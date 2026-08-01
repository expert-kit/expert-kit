"""Platform-specific expert selection behind one remote-runner contract."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.platforms import current_platform

type RoutingFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class ExpertSelector(Protocol):
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ExpertSelectionOptions(Protocol):
    moe_config: FusedMoEConfig
    top_k: int
    use_grouped_topk: bool
    renormalize: bool
    topk_group: int | None
    num_expert_group: int | None
    custom_routing_function: RoutingFunction | None
    scoring_func: str
    routed_scaling_factor: float
    e_score_correction_bias: torch.Tensor | None


class _AscendSelectExperts(Protocol):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: RoutingFunction | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        indices_type: torch.dtype | None = None,
        mix_placement: bool = False,
        num_logical_experts: int = -1,
        num_shared_experts: int = 0,
        num_experts: int = -1,
        input_ids: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _load_ascend_select_experts() -> _AscendSelectExperts:
    try:
        module = import_module("vllm_ascend.ops.fused_moe.experts_selector")
    except ImportError as error:
        raise RuntimeError(
            "vLLM selected the NPU platform but vLLM Ascend expert selection is unavailable"
        ) from error
    select_experts = getattr(module, "select_experts", None)
    if not callable(select_experts):
        raise RuntimeError("vLLM Ascend does not export a callable select_experts")
    return cast(_AscendSelectExperts, select_experts)


class VllmExpertSelector:
    def __init__(self, router: FusedMoERouter) -> None:
        self._router = router

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._router.select_experts(
            hidden_states,
            router_logits,
            topk_indices_dtype=torch.int32,
            input_ids=input_ids,
        )


class AscendExpertSelector:
    def __init__(
        self,
        options: ExpertSelectionOptions,
        *,
        tid2eid: torch.Tensor | None = None,
    ) -> None:
        self._options = options
        self._tid2eid = tid2eid
        self._select = _load_ascend_select_experts()

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        options = self._options
        config = options.moe_config
        return self._select(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=options.top_k,
            use_grouped_topk=options.use_grouped_topk,
            renormalize=options.renormalize,
            topk_group=options.topk_group,
            num_expert_group=options.num_expert_group,
            custom_routing_function=options.custom_routing_function,
            scoring_func=options.scoring_func,
            routed_scaling_factor=options.routed_scaling_factor,
            e_score_correction_bias=options.e_score_correction_bias,
            indices_type=torch.int32,
            mix_placement=False,
            num_logical_experts=config.num_logical_experts,
            num_shared_experts=0,
            num_experts=config.num_experts,
            input_ids=input_ids,
            tid2eid=self._tid2eid,
        )


def create_expert_selector(
    router: FusedMoERouter,
    options: ExpertSelectionOptions,
    *,
    tid2eid: torch.Tensor | None = None,
) -> ExpertSelector:
    if current_platform.device_type == "npu":
        return AscendExpertSelector(options, tid2eid=tid2eid)
    return VllmExpertSelector(router)
