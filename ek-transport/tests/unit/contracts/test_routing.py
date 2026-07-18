"""Tests for Frontend routing validation before wire-dtype conversion."""

from __future__ import annotations

import pytest
import torch

from expertkit_transport.contracts import validate_and_convert_routing


@pytest.mark.parametrize(
    "expert_ids",
    [
        torch.tensor([[2**32, 0]], dtype=torch.int64),
        torch.tensor([[-2, 0]], dtype=torch.int64),
        torch.tensor([[4, 0]], dtype=torch.int64),
    ],
)
def test_rejects_out_of_range_expert_ids(expert_ids: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="configured expert range"):
        validate_and_convert_routing(
            expert_ids,
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            experts_per_layer=4,
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bool],
)
def test_rejects_non_integer_expert_ids(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        validate_and_convert_routing(
            torch.tensor([[0, 1]], dtype=dtype),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            experts_per_layer=4,
        )


def test_rejects_nonzero_weight_for_invalid_assignment() -> None:
    with pytest.raises(ValueError, match="zero routing weight"):
        validate_and_convert_routing(
            torch.tensor([[-1, 1]], dtype=torch.int64),
            torch.tensor([[0.25, 0.75]], dtype=torch.float64),
            experts_per_layer=4,
        )


def test_converts_valid_int64_and_returns_sorted_distinct_experts() -> None:
    expert_ids, routing_weights, distinct_expert_ids = validate_and_convert_routing(
        torch.tensor([[-1, 3], [1, 3]], dtype=torch.int64),
        torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=torch.float16),
        experts_per_layer=4,
    )

    assert expert_ids.dtype is torch.int32
    assert expert_ids.tolist() == [[-1, 3], [1, 3]]
    assert routing_weights.dtype is torch.float32
    assert routing_weights.tolist() == [[0.0, 1.0], [0.25, 0.75]]
    assert distinct_expert_ids == (1, 3)


def test_returns_an_empty_distinct_list_when_every_assignment_is_invalid() -> None:
    _, _, distinct_expert_ids = validate_and_convert_routing(
        torch.tensor([[-1, -1]], dtype=torch.int32),
        torch.zeros((1, 2), dtype=torch.float32),
        experts_per_layer=4,
    )

    assert distinct_expert_ids == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_validation_returns_host_metadata_without_changing_device() -> None:
    expert_ids, routing_weights, distinct_expert_ids = validate_and_convert_routing(
        torch.tensor([[3, 1], [1, 2]], dtype=torch.int64, device="cuda"),
        torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float16, device="cuda"),
        experts_per_layer=4,
    )

    assert expert_ids.device.type == "cuda"
    assert routing_weights.device.type == "cuda"
    assert distinct_expert_ids == (1, 2, 3)
