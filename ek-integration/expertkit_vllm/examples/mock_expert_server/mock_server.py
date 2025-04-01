import grpc
import torch
import io
from concurrent import futures
import time
import logging
import argparse
import json
import numpy as np
from typing import List, Dict, Any, Optional

from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2
from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2_grpc

class ExpertKitServiceMock(expert_pb2_grpc.ComputationServiceServicer):
    """Mock implementation of ExpertKit computation service."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the mock service with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - expert_dim: Dimension of expert outputs
                - experts: Dict mapping expert IDs to their characteristics
                - latency_ms: Optional artificial latency to simulate processing time
        """
        self.config = config
        self.expert_dim = config.get("expert_dim", 4096)
        self.latency_ms = config.get("latency_ms", 0)
        self.experts = config.get("experts", {})
        self.logger = logging.getLogger("ExpertKitMock")
        
        # Initialize fake expert weights (for demonstration)
        self.expert_weights = {}
        for expert_id in self.experts:
            # Create deterministic but unique weights for each expert
            seed = int(hash(expert_id) % 2**32)
            np.random.seed(seed)
            # Create a mock embedding for this expert
            self.expert_weights[expert_id] = torch.from_numpy(
                np.random.normal(0, 0.02, (self.expert_dim, self.expert_dim))
            ).float()
        
        self.logger.info(f"Initialized mock server with {len(self.experts)} experts")
    
    def Forward(self, request, context):
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
            # Deserialize input tensor
            input_buf = io.BytesIO(request.tensor)
            hidden_states = torch.load(input_buf)
            
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
                seq_experts = seq_info.experts
                self.logger.info(f"Sequence {seq_idx} requests {len(seq_experts)} experts: {seq_experts}")
                
                # Process each expert for this sequence
                expert_outputs = []
                
                for expert_id in seq_experts:
                    if expert_id in self.expert_weights:
                        # Apply mock expert computation (simple matrix multiply)
                        input_vector = hidden_states[seq_idx].view(-1, hidden_dim)
                        output = torch.matmul(input_vector, self.expert_weights[expert_id])
                        expert_outputs.append(output)
                    else:
                        # If expert not found, create a dummy output
                        self.logger.warning(f"Expert {expert_id} not found, using zeros")
                        expert_outputs.append(torch.zeros(1, self.expert_dim))
                
                # Stack all expert outputs for this sequence
                seq_output = torch.cat([out for out in expert_outputs], dim=0)
                results.append(seq_output)
            
            # Combine results from all sequences [seq, expert, dim]
            all_results = torch.stack(results)
            
            # Serialize output tensor
            output_buf = io.BytesIO()
            torch.save(all_results, output_buf)
            output_bytes = output_buf.getvalue()
            
            self.logger.info(f"Output tensor shape: {all_results.shape}")
            
            # Return response
            return expert_pb2.ForwardResp(output_tensor=output_bytes)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return expert_pb2.ForwardResp()

def serve(config_path: str, port: int):
    """Start the gRPC server.
    
    Args:
        config_path: Path to configuration JSON file
        port: Port to listen on
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Server")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add servicers
    expert_kit_service = ExpertKitServiceMock(config)
    
    expert_pb2_grpc.add_ComputationServiceServicer_to_server(
        expert_kit_service, server
    )
    
    # Start server
    server_addr = f'[::]:{port}'
    server.add_insecure_port(server_addr)
    server.start()
    
    logger.info(f"Server started, listening on {server_addr}")
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
        "--config", 
        type=str, 
        default="config.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=50051,
        help="Port to listen on"
    )
    
    args = parser.parse_args()
    serve(args.config, args.port)