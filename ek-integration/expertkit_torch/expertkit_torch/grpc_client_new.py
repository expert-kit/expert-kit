import logging
from typing import List
import torch
import safetensors.torch

logger = logging.getLogger(__name__)

try:
    from expertkit_transport import (
        ExpertKitClient as RustExpertKitClient,
        flush_tracing as _flush_tracing,
    )
    RUST_CLIENT_AVAILABLE = True
except ImportError:
    RustExpertKitClient = None
    _flush_tracing = None
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
        self._closed = False
        self.rust_client.connect()

        logger.info(
            f"ExpertKitClient connected: controller={controller_addr}, timeout={timeout_sec}s")

    def forward_expert(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward computation to experts.

        Passes tensor pointer directly to Rust.

        Args:
            expert_ids: Expert IDs for each sequence [batch_size, n_routed_experts]
            hidden_state: Input tensor [batch_size, hidden_dim] (CPU or CUDA)

        Returns:
            Output tensor [batch_size, n_routed_experts, expert_dim] (same device)
        """
        logger.debug(
            f"Sending batch_size={len(expert_ids)}"
        )
        if self._closed or self.rust_client is None:
            raise RuntimeError("ExpertKitClient is closed")

        # Pass tensor directly to Rust
        # Rust accesses tensor memory directly via pointer
        output = self.rust_client.forward_expert(expert_ids, hidden_state)

        logger.debug(f"Received output shape: {output.shape}")

        return output

    def close(self) -> None:
        """Flush frontend tracing spans before releasing the Rust client."""
        if getattr(self, "_closed", True):
            return

        try:
            rust_client = getattr(self, "rust_client", None)
            if rust_client is not None and hasattr(rust_client, "close"):
                rust_client.close()
            elif _flush_tracing is not None:
                _flush_tracing()
        finally:
            self.rust_client = None
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
