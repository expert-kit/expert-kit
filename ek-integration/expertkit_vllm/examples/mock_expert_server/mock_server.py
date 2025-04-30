import grpc
import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import safetensors
from concurrent import futures
import time
import logging
import argparse
import numpy as np
import os
import re
from typing import List, Dict, Any, Optional, Tuple, Literal

from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2
from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2_grpc

MAX_METADATA_SIZE = 20 * 1024  # 20 KB
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MB
torch.set_default_dtype(torch.bfloat16)

class ExpertModule(nn.Module):
    """Expert module implementation matching the Expert class in the model."""
    
    def __init__(self, input_dim: int, expert_dim: int):
        """Initialize the Expert module.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for FFN
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, expert_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing w2(silu(w1(x)) * w3(x))
        
        Args:
            x: Input tensor [hidden_dim]
            
        Returns:
            Output tensor [hidden_dim]
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class ExpertPool(nn.Module):
    """Collection of experts organized by ID."""
    
    def __init__(self, input_dim: int, expert_dim: int):
        """Initialize the expert pool.
        
        Args:
            input_dim: Input dimension for experts
            hidden_dim: Hidden dimension for experts
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = expert_dim
        self.experts = nn.ModuleDict()  # Store experts by ID
    
    def add_expert(self, expert_id: str, expert: ExpertModule):
        """Add an expert to the pool.
        
        Args:
            expert_id: Expert identifier
            expert: Expert module
        """
        self.experts[expert_id] = expert
    
    def forward(self, x: torch.Tensor, expert_id: str) -> torch.Tensor:
        """Run inference with the specified expert.
        
        Args:
            x: Input tensor
            expert_id: ID of expert to use
        
        Returns:
            Output tensor from the expert
        """

        if expert_id in self.experts:
            res = self.experts[expert_id](x)
        else:
            # Return random output for missing experts
            res = torch.randn(x.shape[0], self.input_dim, 
                            device=x.device, dtype=x.dtype)
            
        return res


    @classmethod
    def from_safetensors(cls, file_path: str, input_dim: int, expert_dim: int, logger=None):
        """Create ExpertPool from a safetensors file.
        
        Args:
            file_path: Path to safetensors file
            input_dim: Input dimension for experts
            expert_dim: Hidden dimension for experts
            logger: Optional logger for messages
        
        Returns:
            ExpertPool instance with loaded experts
        """
        expert_pool = cls(input_dim, expert_dim)
        
        if logger:
            logger.info(f"Loading experts from {file_path}")
            
        try:
            # Load weights from file
            if os.path.isfile(file_path) and file_path.endswith('.safetensors'):
                weights = safetensors.torch.load_file(file_path)
                
                # Extract expert weights based on key patterns
                expert_pattern = re.compile(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w\d)\.weight')
                
                # Group weights by layer and expert
                expert_weights = {}
                
                for key, tensor in weights.items():
                    match = expert_pattern.match(key)
                    if match:
                        layer_id, expert_id, weight_type = match.groups()
                        expert_key = f"{layer_id}_{expert_id}"
                        
                        if expert_key not in expert_weights:
                            expert_weights[expert_key] = {}
                        
                        expert_weights[expert_key][weight_type + '.weight'] = tensor
                
                # Create experts from weights
                for expert_key, weights in expert_weights.items():
                    if all(k in weights for k in ['w1.weight', 'w2.weight', 'w3.weight']):
                        expert = ExpertModule(input_dim, expert_dim)
                        
                        # Load weights into expert
                        expert.w1.weight.data.copy_(weights['w1.weight'])
                        expert.w2.weight.data.copy_(weights['w2.weight'])
                        expert.w3.weight.data.copy_(weights['w3.weight'])
                        
                        # Add expert to pool
                        expert_pool.add_expert(expert_key, expert)
                        
                        if logger:
                            logger.info(f"Loaded expert: {expert_key}")
                
                if logger:
                    logger.info(f"Successfully loaded {len(expert_pool.experts)} experts")
            else:
                if logger:
                    logger.error(f"Invalid file path: {file_path}")
        
        except Exception as e:
            if logger:
                logger.error(f"Failed to load weights: {str(e)}")

        return expert_pool


class ExpertKitServiceMock(expert_pb2_grpc.ComputationServiceServicer):
    """Mock implementation of ExpertKit computation service."""
    
    def __init__(
        self, 
        hidden_dim: int,
        expert_dim: int,
        latency_ms: int = 0, 
        weights_path: Optional[str] = None,
        mode: Literal["compute", "zeros"] = "compute"
    ):
        """Initialize the mock service with configuration.
        
        Args:
            expert_dim: Dimension of expert outputs
            hidden_dim: Hidden dimension for FFN
            latency_ms: Optional artificial latency to simulate processing time
            weights_path: Path to directory containing expert weights in safetensors format
            mode: Operation mode - "compute" uses real weights, "zeros" returns zero tensors
        """
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.latency_ms = latency_ms
        self.mode = mode
        self.logger = logging.getLogger("ExpertKitMock")
        
        # Initialize expert pool
        self.expert_pool = None
        
        self.logger.info(f"Initialized mock server")
        self.logger.info(f"Hidden dimension: {self.hidden_dim}")
        self.logger.info(f"Expert dimension: {self.expert_dim}")
        self.logger.info(f"Simulated latency: {self.latency_ms}ms")
        self.logger.info(f"Operation mode: {self.mode}")
        
        # Load expert weights if path provided and mode is compute
        if weights_path and self.mode == "compute":
            self.load_expert_weights(weights_path)
        elif self.mode == "zeros":
            self.logger.info("Running in 'zeros' mode - no weights will be loaded")
    
    def load_expert_weights(self, weights_path: str):
        """Load expert weights from safetensors files.
        
        Args:
            weights_path: Path to directory containing expert weights
        """
        self.logger.info(f"Loading expert weights from: {weights_path}")
        
        if not os.path.exists(weights_path):
            self.logger.error(f"Weights path does not exist: {weights_path}")
            return
        
        try:
            # Determine whether weights_path is a file or directory
            if os.path.isfile(weights_path) and weights_path.endswith('.safetensors'):
                # Create expert pool from single file
                self.expert_pool = ExpertPool.from_safetensors(
                    weights_path, 
                    self.hidden_dim,
                    self.expert_dim, 
                    self.logger
                )
            else:
                self.logger.error(f"Invalid weights path: {weights_path}")
                return
                
            # Create empty expert pool if none was created
            if self.expert_pool is None:
                self.expert_pool = ExpertPool(self.hidden_dim, self.expert_dim)
                
        except Exception as e:
            self.logger.error(f"Failed to load expert weights: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Create empty expert pool if loading failed
            self.expert_pool = ExpertPool(self.hidden_dim, self.expert_dim)
    
    def parse_expert_id(self, expert_id: str) -> Tuple[str, str]:
        """Parse expert ID to extract layer and expert information.
        
        Args:
            expert_id: Expert ID from request (e.g., "3_0", "layer_3_expert_0")
            
        Returns:
            Tuple of normalized expert ID and alternate ID format
        """
        # Handle different potential formats
        # Format 1: "3_0" (layer_expert)
        if re.match(r'^\d+_\d+$', expert_id):
            layer_id, expert_id = expert_id.split('_')
            return f"{layer_id}_{expert_id}", f"layer_{layer_id}_expert_{expert_id}"
        
        # Format 2: "layer_3_expert_0"
        match = re.match(r'^layer_(\d+)_expert_(\d+)$', expert_id)
        if match:
            layer_id, expert_id = match.groups()
            return f"{layer_id}_{expert_id}", expert_id
        
        # Unknown format, return as-is and None
        return expert_id, None
    
    @torch.inference_mode
    def Forward(self, request: expert_pb2.ForwardReq, context):
        """Handle Forward RPC requests.
        
        Args:
            request: The ForwardReq message
            context: gRPC context
            
        Returns:
            ForwardResp with output tensor
        """
        # Log request details
        self.logger.info(f"Received request for instance {request.instance_id}")
        num_sequences = len(request.sequences)
        self.logger.info(f"Processing {num_sequences} sequences")
        
        # Simulate processing time if configured
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
            
        try:
            # Deserialize input tensor - 使用safetensors加载
            hidden_states = safetensors.torch.load(request.tensor)["data"]
            
            # Validate input tensor shape
            batch_size, hidden_dim = hidden_states.shape
            if batch_size != num_sequences:
                error_msg = f"Batch size mismatch: got {batch_size}, expected {num_sequences}"
                self.logger.error(error_msg)
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(error_msg)
                return expert_pb2.ForwardResp()
                
            self.logger.info(f"Input tensor shape: {hidden_states.shape}")
            
            # Process each sequence with its requested experts
            results = []
            
            for seq_idx, seq_info in enumerate(request.sequences):
                num_experts = len(seq_info.experts)
                self.logger.info(f"Sequence {seq_idx} requests {num_experts} experts")
                
                # Get sequence input
                seq_input = hidden_states[seq_idx]
                
                if num_experts > 0:
                    if self.mode == "zeros" or self.expert_pool is None:
                        # Zeros mode - just return zeros in the correct shape
                        self.logger.info(f"Zeros mode: returning zero tensor for {num_experts} experts")
                        seq_output = torch.zeros(num_experts, self.hidden_dim, 
                                               device=hidden_states.device,
                                               dtype=hidden_states.dtype)
                    else:
                        # Compute mode - process with real experts
                        seq_experts_output = []
                        for expert_id in seq_info.experts:
                            self.logger.info(f"Computing with expert: {expert_id}")
                            
                            # Try different expert ID formats
                            normalized_id, alt_id = self.parse_expert_id(expert_id)
                            
                            # Use expert pool for computation
                            if normalized_id in self.expert_pool.experts:
                                expert_output = self.expert_pool.forward(seq_input.unsqueeze(0), normalized_id).squeeze(0)
                            elif alt_id and alt_id in self.expert_pool.experts:
                                expert_output = self.expert_pool.forward(seq_input.unsqueeze(0), alt_id).squeeze(0)
                            else:
                                self.logger.warning(f"Expert {expert_id} not found, using random output")
                                expert_output = torch.randn(self.hidden_dim, device=seq_input.device, 
                                                         dtype=seq_input.dtype)
                            
                            seq_experts_output.append(expert_output)
                        
                        # Stack outputs from all experts for this sequence
                        if seq_experts_output:
                            seq_output = torch.stack(seq_experts_output)
                        else:
                            seq_output = torch.zeros(0, self.hidden_dim, device=hidden_states.device, 
                                                   dtype=hidden_states.dtype)
                else:
                    # Handle case where no experts were requested
                    seq_output = torch.zeros(0, self.hidden_dim, device=hidden_states.device, 
                                            dtype=hidden_states.dtype)
                    
                results.append(seq_output)
            
            # Combine results from all sequences
            if results:
                all_results = torch.stack(results)
            else:
                all_results = torch.zeros(0, 0, self.hidden_dim, device=hidden_states.device, 
                                        dtype=hidden_states.dtype)
            
            self.logger.info(f"Output tensor shape: {all_results.shape}")
            self.logger.info(f"🚀 Output tensor: {all_results}")
            
            # Serialize output tensor
            output_bytes = safetensors.torch.save({"data": all_results})
            
            # Return response
            return expert_pb2.ForwardResp(output_tensor=output_bytes)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(traceback.format_exc())
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return expert_pb2.ForwardResp()

def serve(hidden_dim: int, expert_dim: int, latency_ms: int, port: int, weights_path: Optional[str] = None, mode: str = "compute"):
    """Start the gRPC server.
    
    Args:
        hidden_dim: Hidden dimension for FFN
        expert_dim: Dimension of expert outputs
        latency_ms: Artificial latency in milliseconds
        port: Port to listen on
        weights_path: Optional path to expert weights directory or file
        mode: Operation mode - "compute" uses real weights, "zeros" returns zero tensors
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Server")
    
    # Create server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_metadata_size', MAX_METADATA_SIZE),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ],
    )
    
    # Add servicers
    expert_kit_service = ExpertKitServiceMock(
        hidden_dim=hidden_dim,
        expert_dim=expert_dim,
        latency_ms=latency_ms,
        weights_path=weights_path,
        mode=mode
    )
    
    expert_pb2_grpc.add_ComputationServiceServicer_to_server(
        expert_kit_service, server
    )
    
    # Start server
    server_addr = f'[::]:{port}'
    server.add_insecure_port(server_addr)
    server.start()
    
    logger.info(f"Server started, listening on {server_addr}")
    logger.info(f"Configuration: expert_dim={expert_dim}, hidden_dim={hidden_dim}, latency_ms={latency_ms}, mode={mode}")
    if weights_path and mode == "compute":
        logger.info(f"Using expert weights from: {weights_path}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("Stopping server")
        server.stop(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExpertKit Mock Server")
    parser.add_argument(
        "--expert_dim", 
        type=int, 
        default=32,
        help="Dimension of expert outputs"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=16,  # default to 16
        help="Hidden dimension for FFN in experts"
    )
    parser.add_argument(
        "--latency_ms", 
        type=int, 
        default=0,
        help="Artificial latency in milliseconds to simulate processing time"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=50051,
        help="Port to listen on"
    )
    parser.add_argument(
        "--weights_path", 
        type=str, 
        default=None,
        help="Path to directory or file containing expert weights in safetensors format"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["compute", "zeros"],
        default="compute",
        help="Operation mode: 'compute' for real computation, 'zeros' for zero tensors"
    )
    
    args = parser.parse_args()
    serve(
        expert_dim=args.expert_dim,
        hidden_dim=args.hidden_dim,
        latency_ms=args.latency_ms,
        port=args.port,
        weights_path=args.weights_path,
        mode=args.mode
    )