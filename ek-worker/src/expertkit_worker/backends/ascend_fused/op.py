from __future__ import annotations

from enum import StrEnum

import torch
import torch_npu
from torch import Tensor


class AscendFusedComputeType(StrEnum):
    UNQUANTIZED = "unquantized"
    DYNAMIC_W8A8 = "dynamic_w8a8"


_COMPUTE_TYPE_TO_QUANT_MODE = {
    AscendFusedComputeType.UNQUANTIZED: -1,
    AscendFusedComputeType.DYNAMIC_W8A8: 1,
}


def run_ascend_fused_moe(
    hidden_states: Tensor,
    expert_ids: Tensor,
    expert_weights: Tensor,
    *,
    w13: list[Tensor],
    w2: list[Tensor],
    compute_type: AscendFusedComputeType,
    w13_scale: list[Tensor] | None = None,
    w2_scale: list[Tensor] | None = None,
    swiglu_limit: float = 0.0,
) -> Tensor:
    """Run routed Ascend fused-MoE computation.

    Expected shapes:

        hidden_states:  [T, H]
        expert_ids:     [T, K]
        routing_weights:[T, K]

        w13:            E tensors shaped [H, 2I]
        w2:             E tensors shaped [I, H]

    Dynamic W8A8 additionally requires:

        w13_scale:      E tensors shaped [2I]
        w2_scale:       E tensors shaped [H]

    Expert IDs must already be remapped into the local range [0, E). The
    value E is reserved for inactive routes outside active_expert_range.
    """

    expert_num = len(w13)

    routing_quant_mode = _get_routing_quant_mode(compute_type)
    if routing_quant_mode != -1 and (w13_scale is None or w2_scale is None):
        raise ValueError("dynamic W8A8 requires w13_scale and w2_scale")

    expanded_x, expanded_row_idx, expert_counts, expanded_scale = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        expert_ids,
        active_num=expert_ids.numel(),
        expert_num=expert_num,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=routing_quant_mode,
        active_expert_range=[0, expert_num],
        row_idx_type=0,
    )

    # GMM count-mode expects one token count for each local expert.
    expert_counts = expert_counts.to(dtype=torch.int64)

    if routing_quant_mode == -1:
        expanded_output = _run_unquantized_mlp_ascend_fused(
            expanded_x,
            expert_counts,
            w13=w13,
            w2=w2,
            swiglu_limit=swiglu_limit,
        )
    else:
        assert w13_scale is not None and w2_scale is not None
        expanded_output = _run_dynamic_w8a8_mlp_ascend_fused(
            expanded_x,
            expanded_scale,
            expert_counts,
            w13=w13,
            w13_scale=w13_scale,
            w2=w2,
            w2_scale=w2_scale,
            output_dtype=hidden_states.dtype,
            swiglu_limit=swiglu_limit,
        )

    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=expanded_output,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=expert_weights.to(dtype=expanded_output.dtype),
    )


def _get_routing_quant_mode(compute_type: AscendFusedComputeType) -> int:
    if compute_type not in _COMPUTE_TYPE_TO_QUANT_MODE:
        raise ValueError(f"unsupported Ascend fused compute type: {compute_type}")

    return _COMPUTE_TYPE_TO_QUANT_MODE[compute_type]


def _run_unquantized_mlp_ascend_fused(
    expanded_x: Tensor,
    expert_counts: Tensor,
    *,
    w13: list[Tensor],
    w2: list[Tensor],
    swiglu_limit: float,
) -> Tensor:
    gate_up = torch_npu.npu_grouped_matmul(
        x=[expanded_x],
        weight=w13,
        bias=None,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_counts,
    )[0]

    activated = _apply_swiglu_ascend_fused(gate_up, swiglu_limit=swiglu_limit)

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[activated],
        weight=w2,
        bias=None,
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_counts,
    )[0]

    return hidden_states


def _run_dynamic_w8a8_mlp_ascend_fused(
    expanded_x: Tensor,
    expanded_scale: Tensor,
    expert_counts: Tensor,
    *,
    w13: list[Tensor],
    w13_scale: list[Tensor],
    w2: list[Tensor],
    w2_scale: list[Tensor],
    output_dtype: torch.dtype,
    swiglu_limit: float,
) -> Tensor:
    gate_up = torch_npu.npu_grouped_matmul(
        x=[expanded_x],
        weight=w13,
        bias=None,
        scale=w13_scale,
        per_token_scale=[expanded_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_counts,
        output_dtype=output_dtype,
    )[0]

    activated = _apply_swiglu_ascend_fused(gate_up, swiglu_limit=swiglu_limit)

    quantized_activated, activated_scale = torch_npu.npu_dynamic_quant(
        activated,
        dst_type=torch.int8,
    )

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[quantized_activated],
        weight=w2,
        bias=None,
        scale=w2_scale,
        per_token_scale=[activated_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_counts,
        output_dtype=output_dtype,
    )[0]

    return hidden_states


def _apply_swiglu_ascend_fused(
    gate_up: Tensor,
    *,
    swiglu_limit: float,
) -> Tensor:
    if swiglu_limit > 0:
        gate, up = gate_up.chunk(2, dim=-1)
        gate.clamp_(max=swiglu_limit)
        up.clamp_(min=-swiglu_limit, max=swiglu_limit)

    return torch_npu.npu_swiglu(gate_up, dim=-1)
