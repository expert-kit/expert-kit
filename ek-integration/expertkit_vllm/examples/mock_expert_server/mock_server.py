import grpc
import safetensors.torch
import torch
import io
import safetensors
from concurrent import futures
import time
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional

from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2
from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2_grpc

MAX_METADATA_SIZE = 20 * 1024  # 20 KB
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100 MB

class ExpertKitServiceMock(expert_pb2_grpc.ComputationServiceServicer):
    """Mock implementation of ExpertKit computation service."""
    
    def __init__(self, expert_dim: int, latency_ms: int = 0):
        """Initialize the mock service with configuration.
        
        Args:
            expert_dim: Dimension of expert outputs
            latency_ms: Optional artificial latency to simulate processing time
        """
        self.expert_dim = expert_dim
        self.latency_ms = latency_ms
        self.logger = logging.getLogger("ExpertKitMock")
        
        self.logger.info(f"Initialized mock server")
        self.logger.info(f"Expert dimension: {self.expert_dim}")
        self.logger.info(f"Simulated latency: {self.latency_ms}ms")
    
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
            # Deserialize input tensor
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
                
                # Generate deterministic random outputs for requested number of experts
                seed = int(hash(f"{request.instance_id}_{seq_idx}") % 2**32)
                np.random.seed(seed)
                
                if num_experts > 0:
                    # Create random outputs of appropriate shape for all experts at once
                    seq_output = torch.from_numpy(
                        np.random.normal(0, 0.1, (num_experts, self.expert_dim))
                    ).float()
                else:
                    # Handle case where no experts were requested
                    seq_output = torch.zeros(0, self.expert_dim)
                    
                results.append(seq_output)
            
            # Combine results from all sequences
            all_results = torch.stack(results) if results else torch.zeros(0, 0, self.expert_dim)
            
            # Serialize output tensor
            output_bytes = safetensors.torch.save({"data": all_results})
            
            self.logger.info(f"Output tensor shape: {all_results.shape}")
            
            # Return response
            return expert_pb2.ForwardResp(output_tensor=output_bytes)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            self.logger.error(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(error_msg)
            return expert_pb2.ForwardResp()

def serve(expert_dim: int, latency_ms: int, port: int):
    """Start the gRPC server.
    
    Args:
        expert_dim: Dimension of expert outputs
        latency_ms: Artificial latency in milliseconds
        port: Port to listen on
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
        expert_dim=expert_dim,
        latency_ms=latency_ms
    )
    
    expert_pb2_grpc.add_ComputationServiceServicer_to_server(
        expert_kit_service, server
    )
    
    # Start server
    server_addr = f'[::]:{port}'
    server.add_insecure_port(server_addr)
    server.start()
    
    logger.info(f"Server started, listening on {server_addr}")
    logger.info(f"Configuration: expert_dim={expert_dim}, latency_ms={latency_ms}")
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
        default=7168,
        help="Dimension of expert outputs"
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
    
    args = parser.parse_args()
    serve(
        expert_dim=args.expert_dim,
        latency_ms=args.latency_ms,
        port=args.port
    )