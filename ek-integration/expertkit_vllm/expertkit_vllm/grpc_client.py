import grpc
import os
import torch

# import safetensors
import safetensors.torch as st

from expertkit_vllm.pbpy.ek.worker.v1 import expert_pb2_grpc, expert_pb2
from typing import List

if os.environ.get("EK_WITH_VLLM_MINDSPORE") == "1":
    import sys
    from typing import Any, Dict
    import mindspore as ms
    from mindspore import nn, Parameter, Tensor as Tensor
    import numpy as np
    from safetensors import deserialize, serialize
else:
    from torch import Tensor as Tensor

MAX_METADATA_SIZE = 20 * 1024  # 20 KB
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 100 MB

if os.environ.get("EK_WITH_VLLM_MINDSPORE") == "1":
    _TYPE = {
        "F64": ms.float64,
        "F32": ms.float32,
        "F16": ms.float16,
        "BF16": ms.bfloat16,
        "I64": ms.int64,
        "U64": ms.uint64,
        "I32": ms.int32,
        "U32": ms.uint32,
        "I16": ms.int16,
        "U16": ms.uint16,
        "I8": ms.int8,
        "U8": ms.uint8,
    }


class ExpertKitClient:
    def __init__(self, expertkit_addr: str = "", timeout_sec: float = 2.0):
        """Initialize ExpertKit gRPC client with configurable timeout.

        Args:
            expertkit_addr: Address of the ExpertKit service (host:port)
            timeout_sec: gRPC timeout in seconds (default: 2.0s)
        """
        print(f"🚀 ExpertKitClient Init: ek_addr({expertkit_addr}), timeout({timeout_sec}s)")
        self.channel = grpc.insecure_channel(
            expertkit_addr,
            options=[
                ("grpc.max_metadata_size", MAX_METADATA_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
        self.stub = expert_pb2_grpc.ComputationServiceStub(self.channel)
        self.timeout = timeout_sec

    if os.environ.get("EK_WITH_VLLM_MINDSPORE") == "1":
        # Since Mindspore does not support methods such as tensor.data_ptr and
        # torch.frombuffer, the save and load of safetensors cannot operate
        # correctly. Therefore, the following are the modified function to use
        # safetensors format.
        @staticmethod
        def _flatten(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
            out = {
                k: {
                    "dtype": str(v.dtype).split(".")[-1].lower(),
                    "shape": v.shape,
                    # TODO: use mindspore unrelease api, change to stable
                    "data": v.get_bytes(),
                }
                for k, v in tensors.items()
            }
            return out

        @staticmethod
        def _view2torch(safeview) -> Dict[str, Tensor]:
            result = {}
            for k, v in safeview:
                if len(v["data"]) == 0:
                    exit("No Data found from Expertkit controller!")
                # TODO: use mindspore unrelease api, change to stable
                t = Tensor.convert_bytes_to_tensor(
                    bytes(v["data"]), torch.Size(v["shape"]), _TYPE[v["dtype"]]
                )
                # Need to translate again to get right class
                t = ms.Tensor(t)
                if sys.byteorder == "big":
                    exit("Data byteorder big is no supported yet!")
                result[k] = t
            return result

    def forward_expert(
        self, expert_ids: List[List[str]], hidden_state: Tensor
    ) -> Tensor:
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
        # buf = io.BytesIO()
        # torch.save(hidden_state, buf)
        # tensor_data = buf.getvalue()
        origin_device = hidden_state.device

        if os.environ.get("EK_WITH_VLLM_MINDSPORE") == "1":
            tensor_data = serialize(
                ExpertKitClient._flatten({"data": hidden_state})
            )
        else:
            tensor_data = st.save({"data": hidden_state})

        # Generate expert ids info
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

            if os.environ.get("EK_WITH_VLLM_MINDSPORE") == "1":
                output_tensor = ExpertKitClient._view2torch(
                    deserialize(response.output_tensor)
                )["data"]
                res = output_tensor.move_to(origin_device.type)
            else:
                res = st.load(
                    response.output_tensor,
                )[
                    "data"
                ].to(origin_device)
            return res
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC failed: {e.code().name}") from e
        except (IOError, RuntimeError) as e:
            raise RuntimeError(f"Tensor deserialization failed: {str(e)}") from e
