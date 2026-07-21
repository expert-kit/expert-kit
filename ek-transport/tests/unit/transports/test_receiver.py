"""Tests for Worker-side Transport position contracts."""

import pytest
import torch

from expertkit_transport.transports.base import WorkerPositionSpec


def test_worker_position_spec_normalizes_device() -> None:
    spec = WorkerPositionSpec(
        max_batch_tokens=8,
        hidden_dim=16,
        top_k=2,
        dtype=torch.bfloat16,
        device="cuda:3",
    )

    assert spec.device == torch.device("cuda:3")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("max_batch_tokens", 0, "positive integer"),
        ("hidden_dim", True, "positive integer"),
        ("top_k", -1, "positive integer"),
        ("dtype", torch.int32, "FP16, BF16, or FP32"),
        ("device", "meta", "must be CPU or CUDA"),
    ],
)
def test_worker_position_spec_rejects_invalid_values(
    field: str,
    value: object,
    match: str,
) -> None:
    fields = {
        "max_batch_tokens": 8,
        "hidden_dim": 16,
        "top_k": 2,
        "dtype": torch.float16,
        "device": "cpu",
    }
    fields[field] = value

    with pytest.raises(ValueError, match=match):
        WorkerPositionSpec(**fields)  # type: ignore[arg-type]
