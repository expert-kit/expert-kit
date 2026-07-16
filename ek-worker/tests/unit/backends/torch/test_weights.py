"""Tests for computation-ready Torch expert weights."""

from __future__ import annotations

import pytest
import torch

from expertkit_worker.backends.torch import TorchExpertWeights


def make_weights(dtype: torch.dtype = torch.float32) -> TorchExpertWeights:
    """Return one valid 3-to-5-to-3 expert."""

    return TorchExpertWeights(
        gate_proj=torch.ones(5, 3, dtype=dtype),
        up_proj=torch.ones(5, 3, dtype=dtype),
        down_proj=torch.ones(3, 5, dtype=dtype),
    )


def test_ready_weights_expose_shape_dtype_device_and_storage() -> None:
    weights = make_weights(torch.bfloat16)

    assert weights.hidden_dim == 3
    assert weights.intermediate_dim == 5
    assert weights.dtype is torch.bfloat16
    assert weights.device == torch.device("cpu")
    assert weights.storage_bytes == (5 * 3 + 5 * 3 + 3 * 5) * 2


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("gate_proj", torch.ones(5, 3, dtype=torch.int32), "FP16, BF16, or FP32"),
        ("up_proj", torch.ones(4, 3), "gate and up"),
        ("down_proj", torch.ones(4, 5), "down projection"),
        ("gate_proj", torch.ones(5, 6)[:, ::2], "contiguous"),
    ],
)
def test_ready_weights_reject_invalid_tensor_metadata(
    field: str,
    value: torch.Tensor,
    match: str,
) -> None:
    tensors = {
        "gate_proj": torch.ones(5, 3),
        "up_proj": torch.ones(5, 3),
        "down_proj": torch.ones(3, 5),
    }
    tensors[field] = value

    with pytest.raises(ValueError, match=match):
        TorchExpertWeights(**tensors)


def test_ready_weights_reject_autograd_storage() -> None:
    with pytest.raises(ValueError, match="must not require gradients"):
        TorchExpertWeights(
            gate_proj=torch.ones(5, 3, requires_grad=True),
            up_proj=torch.ones(5, 3),
            down_proj=torch.ones(3, 5),
        )
