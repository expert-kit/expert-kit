"""Tests for Worker-side Transport buffer contracts."""

import pytest
import torch

from expertkit_transport.transports.base import BatchBufferConfig


def test_batch_buffer_config_preserves_device() -> None:
    device = torch.device("cuda:3")
    spec = BatchBufferConfig(
        max_batch_tokens=8,
        hidden_dim=16,
        top_k=2,
        dtype=torch.bfloat16,
        device=device,
    )

    assert spec.device is device


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("max_batch_tokens", 0, "positive integer"),
        ("hidden_dim", True, "positive integer"),
        ("top_k", -1, "positive integer"),
        ("dtype", torch.int32, "FP16, BF16, or FP32"),
        ("device", torch.device("meta"), "must be CPU, CUDA, or NPU"),
    ],
)
def test_batch_buffer_config_rejects_invalid_values(
    field: str,
    value: object,
    match: str,
) -> None:
    fields = {
        "max_batch_tokens": 8,
        "hidden_dim": 16,
        "top_k": 2,
        "dtype": torch.float16,
        "device": torch.device("cpu"),
    }
    fields[field] = value

    with pytest.raises(ValueError, match=match):
        BatchBufferConfig(**fields)  # type: ignore[arg-type]


def test_batch_buffer_config_rejects_string_device() -> None:
    with pytest.raises(TypeError, match=r"must be a torch\.device"):
        BatchBufferConfig(
            max_batch_tokens=8,
            hidden_dim=16,
            top_k=2,
            dtype=torch.float16,
            device="cpu",  # type: ignore[arg-type]
        )
