import grpc
import torch
import io
from expertkit_vllm.pbpy import expert_pb2_grpc, expert_pb2


class ExpertKitClient:
    def __init__(self, expertkit_addr: str, timeout_sec: float = 2.0):
        """Initialize ExpertKit gRPC client with configurable timeout.

        Args:
            expertkit_addr: Address of the ExpertKit service (host:port)
            timeout_sec: gRPC timeout in seconds (default: 2.0s)
        """
        self.channel = grpc.insecure_channel(expertkit_addr)
        self.stub = expert_pb2_grpc.ExpertComputationStub(self.channel)
        self.timeout = timeout_sec

    def forward_expert(
        self,
        layer: int,
        idx: int,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Blocking call to expert-kit. Raises on any failure.

        Args:
            layer: Layer ID 
            idx: Expert index
            hidden_state: Input tensor

        Returns:
            Output tensor from remote expert computation

        Raises:
            RuntimeError: On any gRPC or tensor serialization failure
        """
        # Get batch size from input tensor
        batch_size = hidden_state.size(0)

        # Serialize tensor (no compression)
        buf = io.BytesIO()
        torch.save(hidden_state, buf)
        tensor_data = buf.getvalue()

        try:
            response = self.stub.Forward(
                expert_pb2.ExpertForwardRequest(
                    layer=layer,
                    idx=idx,
                    batch_size=batch_size,
                    tensor=tensor_data
                ),
                timeout=self.timeout
            )
            return torch.load(io.BytesIO(response.output_tensor))
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC failed: {e.code().name}") from e
        except (IOError, RuntimeError) as e:
            raise RuntimeError(f"Tensor serialization failed: {str(e)}") from e
