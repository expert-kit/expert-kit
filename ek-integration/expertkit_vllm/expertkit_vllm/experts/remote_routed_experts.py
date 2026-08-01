"""No-local-weight routed-expert placeholder for vLLM's MoE factory."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

type RoutingFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]


class RemoteRoutedExperts(nn.Module):
    """Carry routing metadata while making routed checkpoint weights absent."""

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        expert_map_manager: ExpertMapManager,
        ckpt_gate_proj_name: str = "gate_proj",
        ckpt_down_proj_name: str = "down_proj",
        ckpt_up_proj_name: str = "up_proj",
        *,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: RoutingFunction | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.params_dtype = params_dtype
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.expert_map_manager = expert_map_manager
        self.ckpt_gate_proj_name = ckpt_gate_proj_name
        self.ckpt_down_proj_name = ckpt_down_proj_name
        self.ckpt_up_proj_name = ckpt_up_proj_name
        self.top_k = moe_config.experts_per_token
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias

        vllm_config = get_current_vllm_config()

        unsupported = {
            "quantization": quant_config is not None,
            "prefill_context_parallel": moe_config.pcp_size != 1,
            "sequence_parallel": moe_config.is_sequence_parallel,
            # The wrapper forces the injected MoE's parallel sizes to one so
            # each vLLM DP rank delegates independently to EK. Read native
            # TP/EP settings from the global config because `moe_config` no
            # longer preserves them.
            "tensor_parallel": vllm_config.parallel_config.tensor_parallel_size != 1,
            "expert_parallel": vllm_config.parallel_config.enable_expert_parallel,
            "eplb": vllm_config.parallel_config.enable_eplb,
            "expert_bias": moe_config.has_bias,
            "fused_shared_experts": expert_map_manager.num_fused_shared_experts != 0,
            "custom_swiglu": any(
                value is not None for value in (swiglu_limit, swiglu_alpha, swiglu_beta)
            ),
            "router_weight_on_input": apply_router_weight_on_input,
            "activation": moe_config.activation is not MoEActivation.SILU,
        }
        enabled = sorted(name for name, value in unsupported.items() if value)
        if enabled:
            raise ValueError(f"Expert Kit remote MoE does not support: {', '.join(enabled)}")

        self._load_weight_sink()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise AssertionError("RemoteRoutedExperts must be executed through RemoteMoERunner")

    def _load_weight_sink(self) -> None:
        for name in ("w13_weight", "w2_weight"):
            param = nn.Parameter(torch.empty(0), requires_grad=False)
            # 1. Weight loader is required by vLLM fused moe
            # when loading expert weights. Expert Kit should discard weights
            # and delegate it to workers
            # 2. `set_weight_attrs` essentially does param.weight_loader = _discard_weight
            # but in a dynamic way
            set_weight_attrs(param, {"weight_loader": _discard_weight})
            self.register_parameter(name, param)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Consume remote expert checkpoint entries without retaining local weights."""
        consumed = False

        # Consume iterator
        for _ in weights:
            consumed = True

        return {"w13_weight", "w2_weight"} if consumed else set()


def _discard_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    *,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    del param, loaded_weight, weight_name, shard_id, expert_id
    return True if return_success else None
