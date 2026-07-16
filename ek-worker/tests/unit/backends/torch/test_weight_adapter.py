"""Tests for zero-copy Torch weight parsing and final placement."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load as official_load
from safetensors.torch import save as official_save

from expertkit_worker.backends.torch import TorchWeightAdapter
from expertkit_worker.weights import parse_safetensors

_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 3


def make_source(dtype: torch.dtype, *, short_names: bool = False) -> tuple[bytearray, dict]:
    """Return an owned expert file and its independent Tensor values."""

    names = (
        ("w1.weight", "w3.weight", "w2.weight")
        if short_names
        else ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    )
    tensors = {
        f"model.expert.{names[0]}": torch.full((_INTERMEDIATE_DIM, _HIDDEN_DIM), 1, dtype=dtype),
        f"model.expert.{names[1]}": torch.full((_INTERMEDIATE_DIM, _HIDDEN_DIM), 3, dtype=dtype),
        f"model.expert.{names[2]}": torch.full((_HIDDEN_DIM, _INTERMEDIATE_DIM), 2, dtype=dtype),
    }
    return bytearray(official_save(tensors)), tensors


def make_adapter(
    source_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: str = "cpu",
) -> TorchWeightAdapter:
    """Return the configured Torch fixture adapter."""

    return TorchWeightAdapter(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=source_dtype,
        compute_dtype=compute_dtype,
        device=device,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_torch_adapter_builds_zero_copy_cpu_views(dtype: torch.dtype) -> None:
    owned, _ = make_source(dtype)
    parsed = parse_safetensors(owned)
    adapter = make_adapter(dtype, dtype)

    cpu_weight = adapter.make_cpu_weight(parsed)
    cpu_weight.gate_proj[0, 0] = 9

    oracle = official_load(bytes(owned))
    assert oracle["model.expert.gate_proj.weight"][0, 0].item() == 9
    ready = adapter.make_ready_weight(cpu_weight)
    assert ready.gate_proj.data_ptr() == cpu_weight.gate_proj.data_ptr()
    assert adapter.cpu_extra_bytes() == 0
    assert adapter.source_tensor_bytes() == 3 * _HIDDEN_DIM * _INTERMEDIATE_DIM * dtype.itemsize
    assert adapter.conversion_temporary_bytes() == 0


def test_torch_adapter_uses_standard_w1_gate_w3_up_mapping() -> None:
    owned, _ = make_source(torch.float32, short_names=True)
    adapter = make_adapter(torch.float32, torch.float32)

    weight = adapter.make_cpu_weight(parse_safetensors(owned))

    torch.testing.assert_close(weight.gate_proj, torch.ones_like(weight.gate_proj))
    torch.testing.assert_close(weight.up_proj, torch.full_like(weight.up_proj, 3))
    torch.testing.assert_close(weight.down_proj, torch.full_like(weight.down_proj, 2))


def test_torch_adapter_converts_only_during_ready_weight_creation() -> None:
    owned, _ = make_source(torch.float32)
    adapter = make_adapter(torch.float32, torch.bfloat16)

    cpu_weight = adapter.make_cpu_weight(parse_safetensors(owned))
    ready = adapter.make_ready_weight(cpu_weight)

    assert cpu_weight.dtype is torch.float32
    assert ready.dtype is torch.bfloat16
    assert ready.gate_proj.data_ptr() != cpu_weight.gate_proj.data_ptr()
    expected_bytes = 3 * _HIDDEN_DIM * _INTERMEDIATE_DIM * 2
    assert adapter.ready_weight_bytes() == expected_bytes


def test_torch_adapter_rejects_wrong_source_metadata() -> None:
    owned, _ = make_source(torch.float16)
    parsed = parse_safetensors(owned)

    with pytest.raises(ValueError, match="unexpected dtype"):
        make_adapter(torch.float32, torch.float32).make_cpu_weight(parsed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_adapter_places_final_weight_on_configured_cuda_device() -> None:
    owned, _ = make_source(torch.float32)
    adapter = make_adapter(torch.float32, torch.float16, "cuda:0")

    ready = adapter.make_ready_weight(adapter.make_cpu_weight(parse_safetensors(owned)))

    assert ready.device == torch.device("cuda:0")
    assert ready.dtype is torch.float16
