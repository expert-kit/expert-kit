"""vLLM 0.25.1 MoE runner that delegates routed experts to Expert Kit."""

from __future__ import annotations

import atexit
import logging
import re
import threading
import weakref
from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import torch
from expertkit_transport import BlockingRoutedMoEClient, validate_and_convert_routing
from torch import nn
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.utils.torch_utils import (
    LayerName,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from expertkit_vllm.experts.remote_routed_experts import RemoteRoutedExperts
from expertkit_vllm.experts.selector import ExpertSelector, create_expert_selector
from expertkit_vllm.utils.config import collect_ek_client_config

logger = logging.getLogger(__name__)

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_CLIENTS: dict[tuple[object, ...], BlockingRoutedMoEClient] = {}
_CLIENTS_LOCK = threading.Lock()
_WRAPPED_FACTORIES: weakref.WeakSet[object] = weakref.WeakSet()

if TYPE_CHECKING:
    from typing import TypeAlias

    # The runtime branch must remain a concrete class for torch schema inference.
    _LayerNameType: TypeAlias = str | LayerName  # noqa: UP040
else:
    _LayerNameType = LayerNameType


def _remote_moe_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    layer_name: _LayerNameType,
) -> torch.Tensor:
    context = get_forward_context()
    layer = context.no_compile_layers[_resolve_layer_name(layer_name)]
    if not isinstance(layer, RemoteMoERunner):
        raise RuntimeError("the vLLM forward context contains the wrong Expert Kit layer")
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
        raise ValueError(f"cannot determine a model layer number from prefix {prefix!r}")
    return int(match.group(1))


def _unwrap_tensor(value: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _num_layers(vllm_config: Any) -> int:
    model_config = vllm_config.model_config
    if model_config is None:
        raise ValueError("vLLM model configuration is unavailable")
    text_config = model_config.hf_text_config
    value = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("the model does not declare a positive num_hidden_layers")
    return value


def _client_for(layer: RemoteMoERunner, hidden_states: torch.Tensor) -> BlockingRoutedMoEClient:
    config = layer.client_config
    key = (
        config.controller_endpoint,
        config.instance_id,
        layer.num_experts,
        layer.num_layers,
        layer.top_k,
        layer.hidden_size,
        hidden_states.dtype,
        hidden_states.device,
    )
    with _CLIENTS_LOCK:
        existing = _CLIENTS.get(key)
        if existing is not None:
            return existing
        client = BlockingRoutedMoEClient(
            config.controller_endpoint,
            instance_id=config.instance_id,
            num_layers=layer.num_layers,
            experts_per_layer=layer.num_experts,
            hidden_dim=layer.hidden_size,
            top_k=layer.top_k,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
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


class RemoteMoERunner(MoERunnerInterface):
    """Preserve vLLM routing and shared experts while offloading routed FFNs."""

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RemoteRoutedExperts,
        enable_dbo: bool = False,
        gate: nn.Module | None = None,
        shared_experts: nn.Module | None = None,
        shared_expert_gate: nn.Module | None = None,
        routed_input_transform: nn.Module | None = None,
        routed_output_transform: nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
        tid2eid: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        unsupported = {
            "dbo": enable_dbo,
            "separate_shared_expert_gate": shared_expert_gate is not None,
            "routed_input_transform": routed_input_transform is not None,
            "routed_output_transform": routed_output_transform is not None,
        }
        enabled = sorted(name for name, value in unsupported.items() if value)
        if enabled:
            raise ValueError(f"Expert Kit remote MoE does not support: {', '.join(enabled)}")

        vllm_config = get_current_vllm_config()
        self.moe_config = moe_config
        self.num_experts = moe_config.num_logical_experts
        self.num_layers = _num_layers(vllm_config)
        self.top_k = moe_config.experts_per_token
        self.hidden_size = moe_config.hidden_dim
        self.layer_name = layer_name
        self._layer_id = _layer_id(layer_name)
        self.router = router
        self.routed_experts = routed_experts
        self.gate = gate
        self._shared_experts_module = shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.expert_selector: ExpertSelector = create_expert_selector(
            router,
            routed_experts,
            tid2eid=tid2eid,
        )
        self.client_config = collect_ek_client_config()

        compilation = vllm_config.compilation_config
        if layer_name in compilation.static_forward_context:
            raise ValueError(f"duplicate Expert Kit MoE layer prefix {layer_name!r}")
        compilation.static_forward_context[layer_name] = self
        compilation.static_all_moe_layers.append(layer_name)
        if compilation.splitting_ops is None:
            compilation.splitting_ops = []
        op_name = "vllm::expertkit_remote_moe"
        if op_name not in compilation.splitting_ops:
            compilation.splitting_ops.append(op_name)
        if compilation.cudagraph_mode.has_full_cudagraphs():
            compilation.cudagraph_mode = CUDAGraphMode.PIECEWISE
            logger.info("disabled full graph capture around remote MoE execution")

    @property
    def is_internal_router(self) -> bool:
        """Return whether this runner owns the model gate."""

        return self.gate is not None

    @property
    def shared_experts(self) -> SharedExperts | None:
        """Expose the vLLM compatibility property without wrapping local experts."""

        return cast(SharedExperts | None, self._shared_experts_module)

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        raise RuntimeError("remote routed experts do not have a local quantization method")

    def _replace_quant_method(self, quant_method: FusedMoEMethodBase) -> None:
        raise RuntimeError("remote routed experts do not have a local quantization method")

    def maybe_init_modular_kernel(self) -> None:
        """No local expert kernel exists in the attention process."""

    @property
    def layer_id(self) -> int:
        return self._layer_id

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def activation(self) -> MoEActivation:
        return self.moe_config.activation

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.routed_experts.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        return None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        return None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        return None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return None

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return None

    def update_expert_map(self) -> None:
        """Reject vLLM EPLB because Controller owns Expert Kit placement."""

        raise RuntimeError("vLLM EPLB is not supported by Expert Kit remote MoE")

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return expert_id

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return ()

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        raise RuntimeError("vLLM EPLB is not supported by Expert Kit remote MoE")

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
        # Expert selector (in vLLM or vLLM-Ascend) will select experts
        # under different platform and runtime
        routing_weights, expert_ids = self.expert_selector.select_experts(
            hidden_states,
            router_logits,
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
        if self._shared_experts_module is not None:
            shared_output = _unwrap_tensor(self._shared_experts_module(hidden_states))
        if self.routed_scaling_factor != 1.0:
            if routed_output.dtype != torch.float16 or shared_output is None:
                routed_output = routed_output * self.routed_scaling_factor
            else:
                shared_output = shared_output / self.routed_scaling_factor
        if shared_output is not None:
            routed_output = routed_output + shared_output
        return routed_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return {f"routed_experts.{name}" for name in self.routed_experts.load_weights(weights)}


def is_expertkit_fused_moe_factory(factory: object) -> bool:
    try:
        return factory in _WRAPPED_FACTORIES
    except TypeError:
        return False


def wrap_fused_moe_factory[**P](
    platform_factory: Callable[P, MoERunnerInterface],
) -> Callable[P, MoERunnerInterface]:
    """Inject remote classes while preserving the active platform wrapper."""

    if is_expertkit_fused_moe_factory(platform_factory):
        return platform_factory

    @wraps(platform_factory)
    def remote_factory(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> MoERunnerInterface:
        options: dict[str, Any] = dict(kwargs)
        conflicts = {
            "custom_runner": options.get("runner_cls") is not None
            or options.get("runner_args") not in (None, {}),
            "custom_experts": options.get("routed_experts_cls") is not None
            or options.get("routed_experts_args") not in (None, {}),
        }
        enabled = sorted(name for name, value in conflicts.items() if value)
        if enabled:
            raise ValueError(f"Expert Kit remote MoE does not support: {', '.join(enabled)}")

        options["runner_cls"] = RemoteMoERunner
        options["routed_experts_cls"] = RemoteRoutedExperts
        # Global DP still controls vLLM scheduling and attention. EK returns a
        # complete routed-expert result for this rank, so the injected MoE must
        # not form vLLM expert collectives across the DP group.
        options["dp_size"] = 1
        options["tp_size"] = 1
        options["pcp_size"] = 1

        invoke = cast(Callable[..., MoERunnerInterface], platform_factory)
        return invoke(*args, **options)

    _WRAPPED_FACTORIES.add(remote_factory)
    return remote_factory
