import logging
from typing import List

import grpc
import torch
import safetensors.torch as st

from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2_grpc, expert_pb2

logger = logging.getLogger(__name__)

# Try to import Rust transport client
try:
    from expertkit_transport import ExpertKitClient as RustExpertKitClient

    RUST_CLIENT_AVAILABLE = True
    print("🚀expertkit-transport available, using Rust transport client")
except ImportError as e:
    RUST_CLIENT_AVAILABLE = False
    RustExpertKitClient = None
    print(f"🚀 {e}")
    print(
        "🚀expertkit-transport not available, using gRPC fallback. "
        "For multi-transport support (gRPC/SHM/RDMA), install with: "
        "cd expertkit-transport-rs && maturin develop -r"
    )

MAX_METADATA_SIZE = 20 * 1024  # 20 KB
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1 GB


class ExpertKitClient:
    def __init__(self, expertkit_addr: str = "", timeout_sec: float = 2.0):
        """Initialize ExpertKit client with configurable timeout.

        Args:
            expertkit_addr: Address of the ExpertKit controller (host:port)
            timeout_sec: Request timeout in seconds (default: 2.0s)
        """
        self.addr = expertkit_addr
        self.timeout = timeout_sec
        self.rust_client = None
        self.stub = None
        self.channel = None

        if RUST_CLIENT_AVAILABLE:
            self._init_rust_client()
        else:
            self._init_grpc_client()

    def _init_rust_client(self):
        """Initialize Rust transport client."""
        self.rust_client = RustExpertKitClient(self.addr, self.timeout)
        self.rust_client.connect()
        print(
            f"ExpertKitClient (transport-rs): addr={self.addr}, timeout={self.timeout}s")

    def _init_grpc_client(self):
        """Initialize pure Python gRPC client (fallback)."""
        self.channel = grpc.insecure_channel(
            self.addr,
            options=[
                ("grpc.max_metadata_size", MAX_METADATA_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
        self.stub = expert_pb2_grpc.ComputationServiceStub(self.channel)
        print(
            f"ExpertKitClient (gRPC fallback): addr={self.addr}, timeout={self.timeout}s")

    def forward_expert(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward computation to experts.

        Args:
            expert_ids: Experts activated for each sequence [batch_size, n_routed_experts]
            hidden_state: Input tensor [batch_size, hidden_dim] (CPU or CUDA)

        Returns:
            Output tensor [batch_size, n_routed_experts, expert_dim] (same device)

        Raises:
            RuntimeError: On any transport or tensor serialization failure
        """
        if self.rust_client is not None:
            return self._forward_p2p(expert_ids, hidden_state)
        else:
            return self._forward_controller(expert_ids, hidden_state)

    def _forward_p2p(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward via Rust transport client (zero-copy)."""
        logger.debug(f"Sending batch_size={len(expert_ids)} via transport-rs")
        output = self.rust_client.forward_expert(expert_ids, hidden_state)
        logger.debug(f"Received output shape: {output.shape}")
        return output

    def _forward_controller(
        self, expert_ids: List[List[str]], hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward via pure Python gRPC (fallback)."""
        origin_device = hidden_state.device
        tensor_data = st.save({"data": hidden_state})

        seq_infos = []
        for ids in expert_ids:
            seq_infos.append(expert_pb2.ForwardReq.SequenceInfo(experts=ids))

        try:
            response: expert_pb2.ForwardResp = self.stub.Forward(
                expert_pb2.ForwardReq(
                    instance_id="test", sequences=seq_infos, tensor=tensor_data
                ),
                timeout=self.timeout,
            )

            return st.load(response.output_tensor)["data"].to(origin_device)
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC failed: {e.code().name}") from e
        except (IOError, RuntimeError) as e:
            raise RuntimeError(f"Tensor serialization failed: {str(e)}") from e

    def refresh_routing(self):
        """Refresh routing table from controller (transport-rs only)"""
        if self.rust_client is not None:
            self.rust_client.refresh_routing()
            logger.debug("Routing table refreshed")
