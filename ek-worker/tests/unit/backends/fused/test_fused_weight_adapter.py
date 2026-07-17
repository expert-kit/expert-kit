"""CUDA tests for fixed fused expert-weight slots."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save as official_save

from expertkit_worker.backends.fused import FusedCpuWeights, FusedWeightAdapter
from expertkit_worker.weights import parse_safetensors

_HIDDEN_DIM = 16
_INTERMEDIATE_DIM = 32


def _cpu_weight(seed: int, dtype: torch.dtype) -> FusedCpuWeights:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(rows, columns, generator=generator, dtype=torch.float32).to(dtype)

    return FusedCpuWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def _adapter(dtype: torch.dtype) -> FusedWeightAdapter:
    return FusedWeightAdapter(
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=dtype,
        compute_dtype=dtype,
        device="cuda:0",
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_adapter_parses_weight_server_safetensors_layout() -> None:
    dtype = torch.float16
    encoded = bytearray(
        official_save(
            {
                "model.expert.gate_proj.weight": torch.ones(
                    _INTERMEDIATE_DIM,
                    _HIDDEN_DIM,
                    dtype=dtype,
                ),
                "model.expert.up_proj.weight": torch.full(
                    (_INTERMEDIATE_DIM, _HIDDEN_DIM),
                    2,
                    dtype=dtype,
                ),
                "model.expert.down_proj.weight": torch.full(
                    (_HIDDEN_DIM, _INTERMEDIATE_DIM),
                    3,
                    dtype=dtype,
                ),
            }
        )
    )
    adapter = _adapter(dtype)
    adapter.initialize_ready_storage(1)

    cpu_weight = adapter.make_cpu_weight(parse_safetensors(encoded))
    ready = adapter.make_ready_weight(cpu_weight, layer_id=1, expert_id=3)

    torch.testing.assert_close(
        ready.storage.gate_up[ready.slot, :_INTERMEDIATE_DIM].cpu(),
        torch.ones_like(cpu_weight.gate_proj),
    )
    torch.testing.assert_close(
        ready.storage.gate_up[ready.slot, _INTERMEDIATE_DIM:].cpu(),
        torch.full_like(cpu_weight.up_proj, 2),
    )
    torch.testing.assert_close(
        ready.storage.down[ready.slot].cpu(),
        torch.full_like(cpu_weight.down_proj, 3),
    )
    adapter.release_ready_weight(ready)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_adapter_preallocates_maps_and_reuses_only_released_slots(
    dtype: torch.dtype,
) -> None:
    adapter = _adapter(dtype)
    adapter.initialize_ready_storage(2)

    first = adapter.make_ready_weight(_cpu_weight(1, dtype), layer_id=0, expert_id=1)
    second = adapter.make_ready_weight(_cpu_weight(2, dtype), layer_id=1, expert_id=2)

    assert first.storage is second.storage
    assert {first.slot, second.slot} == {0, 1}
    assert first.storage.slot_for(0, 1) == first.slot
    assert first.storage.slot_for(1, 2) == second.slot
    assert first.storage.gate_up.shape == (
        2,
        2 * _INTERMEDIATE_DIM,
        _HIDDEN_DIM,
    )
    assert first.storage.down.shape == (2, _HIDDEN_DIM, _INTERMEDIATE_DIM)
    with pytest.raises(RuntimeError, match="no free fused weight slot"):
        adapter.make_ready_weight(_cpu_weight(3, dtype), layer_id=0, expert_id=3)

    adapter.release_ready_weight(first)
    replacement = adapter.make_ready_weight(_cpu_weight(3, dtype), layer_id=0, expert_id=3)

    assert replacement.slot == first.slot
    assert replacement.storage.slot_for(0, 1) == -1
    assert replacement.storage.slot_for(0, 3) == replacement.slot
    adapter.release_ready_weight(second)
    adapter.release_ready_weight(replacement)


def test_fused_adapter_rejects_unsupported_dtype_and_device() -> None:
    options = {
        "num_layers": 2,
        "experts_per_layer": 4,
        "hidden_dim": _HIDDEN_DIM,
        "intermediate_dim": _INTERMEDIATE_DIM,
        "source_dtype": torch.float16,
        "compute_dtype": torch.float16,
    }
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        FusedWeightAdapter(**options, device="cpu")
    with pytest.raises(ValueError, match="FP16 or BF16"):
        FusedWeightAdapter(
            **(options | {"compute_dtype": torch.float32}),
            device="cuda:0",
        )
