from dataclasses import dataclass

import torch
import torch_npu


@dataclass(slots=True, frozen=True)
class AscendFusedMoEOutput:
    output: torch.Tensor


def ascend_fused_moe(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
) -> AscendFusedMoEOutput:
    expert_num = len(w13)

    # Grouping
    expanded_x, expanded_row_idx, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        expert_ids,
        active_num=expert_ids.numel(),
        expert_num=expert_num,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=[0, expert_num],
        row_idx_type=0,
    )

    # Gate & up

    # Down
