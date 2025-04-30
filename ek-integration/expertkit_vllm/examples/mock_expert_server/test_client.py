import torch
import argparse
import logging
from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2_grpc, expert_pb2
from typing import List

from expertkit_vllm.grpc_client import ExpertKitClient

torch.set_default_dtype(torch.bfloat16)

def test_forward(client: ExpertKitClient, batch_size: int = 2, hidden_dim: int = 4096):
    """Test the Forward RPC call.
    
    Args:
        client: ExpertKitClient instance
        batch_size: Number of sequences to send
        hidden_dim: Hidden dimension size
    """
    logger = logging.getLogger("TestClient")
    
    # Create random hidden states
    hidden_states = torch.zeros(batch_size, hidden_dim)
    
    # Create expert IDs (simulating router output)
    expert_ids = []
    for i in range(batch_size):
        # Each sequence gets 2 experts from the same layer (for testing)
        layer_id = i % 3 + 3  # Use layers 0, 1, 2
        expert_ids.append([f"{layer_id}_0", f"{layer_id}_1"])
    
    logger.info(f"Sending request with {batch_size} sequences")
    logger.info(f"Hidden states shape: {hidden_states.shape}")
    logger.info(f"Expert IDs: {expert_ids}")
    
    # Call the client
    result = client.forward_expert(expert_ids, hidden_states)
    
    # Log the result
    logger.info(f"Received response with shape: {result.shape}")
    logger.info(f"Output sample (first sequence, first expert): {result[0, 0]}")
    
    # Basic validation
    expected_shape = (batch_size, 2, hidden_dim)  # [batch_size, num_experts_per_seq, hidden_dim]
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    
    logger.info("Test completed successfully")
    return True
        

def main():
    parser = argparse.ArgumentParser(description="ExpertKit Mock Client Test")
    parser.add_argument(
        "--server", 
        type=str, 
        default="localhost:50051",
        help="Server address in the format host:port"
    )
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=2.0,
        help="gRPC timeout in seconds"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=2,
        help="Number of sequences to send"
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=32,
        help="Hidden dimension size"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create client
    print(f"🚀client addr: {args.server}")
    client = ExpertKitClient(args.server, args.timeout)
    
    # Run test
    success = test_forward(client, args.batch_size, args.hidden_dim)
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed. Check logs for details.")
        exit(1)

if __name__ == "__main__":
    main()