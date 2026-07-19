"""vLLM 0.25.1 MoE runner that delegates routed experts to Expert Kit."""

from __future__ import annotations

import atexit
import logging
import re
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch
from expertkit_transport.adapters.grpc import BlockingGrpcRoutedMoEClient
from expertkit_transport.contracts import validate_and_convert_routing
from expertkit_vllm.utils.config import collect_ek_client_config
from torch import nn
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import (
    LayerName,
    LayerNameType,
    direct_register_custom_op,
)

logger = logging.getLogger(__name__)

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_CLIENTS: dict[tuple[object, ...], BlockingGrpcRoutedMoEClient] = {}
_CLIENTS_LOCK = threading.Lock()

if TYPE_CHECKING:
    from typing import TypeAlias

    # The runtime branch must remain a concrete class for torch schema inference.
    _LayerNameType: TypeAlias = str | LayerName  # noqa: UP040
else:
    _LayerNameType = LayerNameType


def _encode_layer_name(layer_name: str) -> _LayerNameType:
    return LayerName(layer_name) if _LayerNameType is LayerName else layer_name


def _resolve_layer_name(layer_name: _LayerNameType) -> str:
    return layer_name.value if isinstance(layer_name, LayerName) else layer_name


def _remote_moe_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    context = get_forward_context()
    layer = context.no_compile_layers[_resolve_layer_name(layer_name)]
    if not isinstance(layer, RemoteMoERunner):
        raise RuntimeError(
            "the vLLM forward context contains the wrong Expert Kit layer"
        )
    result = layer._forward_impl(hidden_states, router_logits, input_ids)
    hidden_states.copy_(result)
    return hidden_states


def _remote_moe_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="expertkit_remote_moe",
    op_func=_remote_moe_impl,
    mutates_args=["hidden_states"],
    fake_impl=_remote_moe_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _layer_id(prefix: str) -> int:
    match = _LAYER_PATTERN.search(prefix)
    if match is None:
        raise ValueError(
            f"cannot determine a model layer number from prefix {prefix!r}"
        )
    return int(match.group(1))


def _unwrap_tensor(value: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _num_layers() -> int:
    model_config = get_current_vllm_config().model_config
    if model_config is None:
        raise ValueError("vLLM model configuration is unavailable")
    text_config = model_config.hf_text_config
    value = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("the model does not declare a positive num_hidden_layers")
    return value


def _client_for(
    layer: RemoteMoERunner, hidden_states: torch.Tensor
) -> BlockingGrpcRoutedMoEClient:
    config = layer.client_config
    key = (
        config.controller_endpoint,
        config.instance_id,
        config.transport,
        layer.num_experts,
        layer.top_k,
        layer.hidden_size,
        hidden_states.dtype,
        hidden_states.device,
    )
    with _CLIENTS_LOCK:
        existing = _CLIENTS.get(key)
        if existing is not None:
            return existing
        client = BlockingGrpcRoutedMoEClient(
            config.controller_endpoint,
            instance_id=config.instance_id,
            num_layers=_num_layers(),
            experts_per_layer=layer.num_experts,
            hidden_dim=layer.hidden_size,
            top_k=layer.top_k,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            worker_transport=config.transport,
        )
        try:
            client.start(timeout_seconds=config.timeout_seconds)
        except BaseException:
            client.close()
            raise
        _CLIENTS[key] = client
        return client


def close_clients() -> None:
    """Close every process-local Transport client created by vLLM layers."""

    with _CLIENTS_LOCK:
        clients = tuple(_CLIENTS.values())
        _CLIENTS.clear()
    for client in clients:
        client.close()


atexit.register(close_clients)


class RemoteMoERunner(nn.Module):
    """Preserve vLLM routing and shared experts while offloading routed FFNs."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        prefix: str,
        router: FusedMoERouter,
        gate: nn.Module | None,
        shared_experts: nn.Module | None,
        apply_routed_scale_to_output: bool,
        routed_scaling_factor: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.layer_name = prefix
        self.layer_id = _layer_id(prefix)
        self.router = router
        self.gate = gate
        self.shared_experts = shared_experts
        self.apply_routed_scale_to_output = apply_routed_scale_to_output
        self.routed_scaling_factor = routed_scaling_factor
        self.client_config = collect_ek_client_config()

        compilation = get_current_vllm_config().compilation_config
        if prefix in compilation.static_forward_context:
            raise ValueError(f"duplicate Expert Kit MoE layer prefix {prefix!r}")
        compilation.static_forward_context[prefix] = self
        compilation.static_all_moe_layers.append(prefix)
        if compilation.splitting_ops is None:
            compilation.splitting_ops = []
        op_name = "vllm::expertkit_remote_moe"
        if op_name not in compilation.splitting_ops:
            compilation.splitting_ops.append(op_name)
        if compilation.cudagraph_mode.has_full_cudagraphs():
            compilation.cudagraph_mode = CUDAGraphMode.PIECEWISE
            logger.info("disabled full CUDA Graph capture around remote MoE execution")

    @property
    def is_internal_router(self) -> bool:
        """Return whether this runner owns the model gate."""

        return self.gate is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run this layer as an eager split between compiled vLLM segments."""

        return torch.ops.vllm.expertkit_remote_moe(
            hidden_states,
            router_logits,
            input_ids,
            _encode_layer_name(self.layer_name),
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gate is not None:
            router_logits = _unwrap_tensor(self.gate(hidden_states))
        routing_weights, expert_ids = self.router.select_experts(
            hidden_states,
            router_logits,
            topk_indices_dtype=torch.int32,
            input_ids=input_ids,
        )
        expert_ids, routing_weights, distinct_expert_ids = validate_and_convert_routing(
            expert_ids,
            routing_weights,
            experts_per_layer=self.num_experts,
        )
        routed_output = _client_for(self, hidden_states).execute(
            layer_id=self.layer_id,
            hidden_states=hidden_states,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            distinct_expert_ids=distinct_expert_ids,
            timeout_seconds=self.client_config.timeout_seconds,
        )

        shared_output: torch.Tensor | None = None
        if self.shared_experts is not None:
            shared_output = _unwrap_tensor(self.shared_experts(hidden_states))
        if self.apply_routed_scale_to_output and self.routed_scaling_factor != 1.0:
            if routed_output.dtype != torch.float16 or shared_output is None:
                routed_output = routed_output * self.routed_scaling_factor
            else:
                shared_output = shared_output / self.routed_scaling_factor
        if shared_output is not None:
            routed_output = routed_output + shared_output
        return routed_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consume remote expert checkpoint entries without retaining local weights."""

        return {name for name, _ in weights}

    def update_expert_map(self) -> None:
        """Reject vLLM EPLB because Controller owns Expert Kit placement."""

        raise RuntimeError("vLLM EPLB is not supported by Expert Kit remote MoE")


def remote_fused_moe(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    intermediate_pad: int | None = None,
    params_dtype: torch.dtype | None = None,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    quant_config: QuantizationConfig | None = None,
    tp_size: int | None = None,
    dp_size: int | None = None,
    pcp_size: int | None = None,
    prefix: str = "",
    custom_routing_function: Callable | None = None,
    router: FusedMoERouter | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    is_sequence_parallel: bool = False,
    reduce_results: bool = True,
    ckpt_names: tuple[str, str, str] = ("gate_proj", "down_proj", "up_proj"),
    n_shared_experts: int | None = None,
    router_logits_dtype: torch.dtype | None = None,
    gate: nn.Module | None = None,
    shared_experts: nn.Module | None = None,
    shared_expert_gate: nn.Module | None = None,
    routed_input_transform: nn.Module | None = None,
    routed_output_transform: nn.Module | None = None,
    apply_routed_scale_to_output: bool = False,
    zero_expert_type: str | None = None,
    hash_indices_table: torch.Tensor | None = None,
    runner_cls: type[Any] | None = None,
    runner_args: dict[str, Any] | None = None,
    routed_experts_cls: type[Any] | None = None,
    routed_experts_args: dict[str, Any] | None = None,
) -> RemoteMoERunner:
    """Match the vLLM 0.25.1 `FusedMoE` factory for the supported subset."""

    config = get_current_vllm_config()
    unsupported = {
        "quant_config": quant_config is not None,
        "tensor_parallel": config.parallel_config.tensor_parallel_size != 1
        or tp_size not in (None, 1),
        "prefill_context_parallel": config.parallel_config.prefill_context_parallel_size
        != 1
        or pcp_size not in (None, 1),
        "expert_parallel": config.parallel_config.enable_expert_parallel,
        "sequence_parallel": is_sequence_parallel,
        "eplb": enable_eplb or num_redundant_experts != 0,
        "expert_bias": has_bias,
        "fused_shared_experts": n_shared_experts not in (None, 0),
        "separate_shared_expert_gate": shared_expert_gate is not None,
        "custom_swiglu": any(
            value is not None for value in (swiglu_limit, swiglu_alpha, swiglu_beta)
        ),
        "router_weight_on_input": apply_router_weight_on_input,
        "routed_input_transform": routed_input_transform is not None,
        "routed_output_transform": routed_output_transform is not None,
        "zero_expert": zero_expert_type is not None,
        "hash_routing": hash_indices_table is not None,
        "custom_runner": runner_cls is not None or runner_args is not None,
        "custom_experts": routed_experts_cls is not None
        or routed_experts_args is not None,
    }
    enabled = sorted(name for name, value in unsupported.items() if value)
    if enabled:
        raise ValueError(
            f"Expert Kit remote MoE does not support: {', '.join(enabled)}"
        )
    if not reduce_results:
        raise ValueError("Expert Kit remote MoE requires reduce_results=True")
    if activation != "silu":
        raise ValueError("Expert Kit remote MoE initially supports only SiLU experts")

    if router is None:
        router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=num_experts,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=(
                1.0 if apply_routed_scale_to_output else routed_scaling_factor
            ),
            e_score_correction_bias=e_score_correction_bias,
            custom_routing_function=custom_routing_function,
        )

    return RemoteMoERunner(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        prefix=prefix,
        router=router,
        gate=gate,
        shared_experts=shared_experts,
        apply_routed_scale_to_output=apply_routed_scale_to_output,
        routed_scaling_factor=routed_scaling_factor,
    )
