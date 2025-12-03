import logging
from typing import List
import torch
import safetensors.torch

logger = logging.getLogger(__name__)

try:
    from expertkit_transport import ExpertKitClient as RustExpertKitClient
    RUST_CLIENT_AVAILABLE = True
except ImportError:
    RUST_CLIENT_AVAILABLE = False
    logger.warning("Rust client not available - ExpertKitClient will not work")


class ExpertKitClient:
    """
    ExpertKit client - thin wrapper around Rust implementation.

    Simplified API:
        client = ExpertKitClient("127.0.0.1:5002")
        output = client.forward_expert(expert_ids, hidden_state)
    """

    def __init__(self, controller_addr: str, timeout_sec: float = 2.0):
        """
        Initialize and connect ExpertKit client.

        Args:
            controller_addr: Controller address (host:port)
            timeout_sec: Request timeout in seconds
        """
        if not RUST_CLIENT_AVAILABLE:
            raise RuntimeError(
                "Rust client not available. Install with: pip install -e expertkit-transport-rs"
            )

        # Create Rust client and connect immediately
        self.rust_client = RustExpertKitClient(controller_addr, timeout_sec)
        self.rust_client.connect()

        logger.info(
            f"ExpertKitClient connected: controller={controller_addr}, timeout={timeout_sec}s")

    def forward_expert(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward computation to experts.

        All async operations happen in Rust - this is synchronous from Python's perspective.

        Args:
            expert_ids: Expert IDs for each sequence [batch_size, n_routed_experts]
            hidden_state: Input tensor [batch_size, hidden_dim]

        Returns:
            Output tensor [batch_size, n_routed_experts, expert_dim]
        """
        # Decompose requests by expert (as Rust expects)
        expert_to_sequences = {}
        for seq_idx, experts in enumerate(expert_ids):
            for expert_idx, expert_id in enumerate(experts):
                if expert_id not in expert_to_sequences:
                    expert_to_sequences[expert_id] = []
                expert_to_sequences[expert_id].append((seq_idx, expert_idx))

        # Prepare all expert data
        all_expert_ids = []
        all_tensor_data = []
        expert_mapping = []

        for expert_id, seq_positions in expert_to_sequences.items():
            seq_indices = [pos[0] for pos in seq_positions]
            expert_input = hidden_state[seq_indices].contiguous().cpu()

            # Serialize to safetensors
            tensor_bytes = safetensors.torch.save({"data": expert_input})

            all_expert_ids.append(expert_id)
            all_tensor_data.append(tensor_bytes)
            expert_mapping.append(
                (expert_id, seq_positions, expert_input.shape))

        # Send all experts to Rust client (blocks until all complete)
        logger.debug(f"Sending {len(all_expert_ids)} experts to Rust client")
        responses = self.rust_client.send_expert_batch(
            all_expert_ids, all_tensor_data)

        # Reconstruct output in original format
        batch_size = len(expert_ids)
        n_experts_per_seq = len(expert_ids[0]) if expert_ids else 0

        result = [[None for _ in range(n_experts_per_seq)]
                  for _ in range(batch_size)]

        for (expert_id, seq_positions, input_shape), response_bytes in zip(expert_mapping, responses):
            # Deserialize response
            expert_output = safetensors.torch.load(response_bytes)["data"]

            # Place outputs back in correct positions
            for output_idx, (seq_idx, expert_pos) in enumerate(seq_positions):
                result[seq_idx][expert_pos] = expert_output[output_idx]

        # Convert to tensor
        output_tensors = []
        for seq_result in result:
            seq_tensors = [t for t in seq_result if t is not None]
            if len(seq_tensors) != n_experts_per_seq:
                raise RuntimeError(
                    f"Incomplete result: got {len(seq_tensors)}, expected {n_experts_per_seq}"
                )
            output_tensors.append(torch.stack(seq_tensors, dim=0))

        final_output = torch.stack(output_tensors, dim=0)
        return final_output.to(hidden_state.device)
