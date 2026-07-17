"""Validate Frontend router output before narrowing it to wire dtypes."""

from __future__ import annotations

import torch


def validate_and_convert_routing(
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    experts_per_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return validated int32 expert IDs and FP32 routing weights.

    This function reads Tensor values and may synchronize a CUDA device. Call it
    once on final router output, before an integer narrowing conversion could
    change an invalid expert ID into a valid one.
    """

    if (
        isinstance(experts_per_layer, bool)
        or not isinstance(experts_per_layer, int)
        or not 0 < experts_per_layer <= torch.iinfo(torch.int32).max + 1
    ):
        raise ValueError("experts_per_layer must fit the int32 expert-ID range")
    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must have shape [token_count, top_k]")
    if routing_weights.shape != expert_ids.shape:
        raise ValueError("routing_weights must match expert_ids")
    if routing_weights.device != expert_ids.device:
        raise ValueError("routing_weights and expert_ids must use the same device")
    try:
        integer_info = torch.iinfo(expert_ids.dtype)
    except TypeError as error:
        raise ValueError("expert_ids must use an integer dtype") from error

    checked_ids = expert_ids
    if integer_info.min >= 0:
        checked_ids = expert_ids.to(
            torch.float64 if integer_info.max > torch.iinfo(torch.int64).max else torch.int64
        )
    invalid_range = (checked_ids < -1) | (checked_ids >= experts_per_layer)
    invalid_weight = (checked_ids == -1) & (routing_weights != 0)
    invalid_flags = torch.stack((invalid_range.any(), invalid_weight.any()))
    range_error, weight_error = invalid_flags.detach().to(device="cpu").tolist()
    if range_error:
        raise ValueError("expert_ids must be within the configured expert range [-1, count)")
    if weight_error:
        raise ValueError("an expert ID of -1 must have zero routing weight")
    return (
        expert_ids.to(dtype=torch.int32),
        routing_weights.to(dtype=torch.float32),
    )
