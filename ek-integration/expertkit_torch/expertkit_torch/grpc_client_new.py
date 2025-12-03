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

        Serializes input once, Rust handles all decomposition and reconstruction.

        Args:
            expert_ids: Expert IDs for each sequence [batch_size, n_routed_experts]
            hidden_state: Input tensor [batch_size, hidden_dim]

        Returns:
            Output tensor [batch_size, n_routed_experts, expert_dim]
        """
        origin_device = hidden_state.device

        # Serialize input once
        hidden_state_bytes = safetensors.torch.save(
            {"data": hidden_state.cpu().contiguous()}
        )

        logger.debug(
            f"Sending batch_size={len(expert_ids)} to Rust (all processing in Rust)"
        )

        # Rust does everything: decompose, route, dispatch, reconstruct
        response_bytes = self.rust_client.forward_expert(
            expert_ids, hidden_state_bytes
        )

        # Deserialize output once
        output = safetensors.torch.load(response_bytes)["data"]

        return output.to(origin_device)
