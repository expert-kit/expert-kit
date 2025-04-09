import grpc
import torch
import io
from expertkit_torch.pbpy.ek.worker.v1 import expert_pb2_grpc, expert_pb2
from typing import List


class ExpertKitClient:
    def __init__(self, expertkit_addr: str, timeout_sec: float = 2.0):
        """Initialize ExpertKit gRPC client with configurable timeout.

        Args:
            expertkit_addr: Address of the ExpertKit service (host:port)
            timeout_sec: gRPC timeout in seconds (default: 2.0s)
        """
        self.channel = grpc.insecure_channel(expertkit_addr)
        self.stub = expert_pb2_grpc.ComputationServiceStub(self.channel)
        self.timeout = timeout_sec

    def forward_expert(
        self,
        expert_ids: List[List[str]],
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Blocking call to expert-kit. Raises on any failure.

        Args:
            expert_ids: Experts activated for each sequence, shape in [batch_size, n_routed_experts]
            hidden_state: Attention output, shape in [batch_size, attn_dim]

        Returns:
            Output tensor from remote expert computation, shape in [batch_size, n_routed_experts, expert_dim]

        Raises:
            RuntimeError: On any gRPC or tensor serialization failure
        """
        # Serialize tensor (no compression)
        buf = io.BytesIO()
        torch.save(hidden_state, buf)
        tensor_data = buf.getvalue()

        # Generate expert ids info
        seq_infos = []
        for ids in expert_ids:
            seq_infos.append(
                expert_pb2.ForwardReq.SequenceInfo(
                    experts=ids
                )
            )

        try:
            response: expert_pb2.ForwardResp = self.stub.Forward(
                expert_pb2.ForwardReq(
                    instance_id="test",
                    sequences=seq_infos,
                    tensor=tensor_data
                ),
                timeout=self.timeout
            )
            return torch.load(io.BytesIO(response.output_tensor))
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC failed: {e.code().name}") from e
        except (IOError, RuntimeError) as e:
            raise RuntimeError(f"Tensor serialization failed: {str(e)}") from e
