"""Tests for GGML SafeTensors parsing and ready-view construction."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save as official_save

from expertkit_worker.backends.ggml import GgmlWeightAdapter
from expertkit_worker.weights import parse_safetensors

_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 3


def _source(dtype: torch.dtype) -> bytearray:
    return bytearray(
        official_save(
            {
                "model.expert.gate_proj.weight": torch.full(
                    (_INTERMEDIATE_DIM, _HIDDEN_DIM), 1, dtype=dtype
                ),
                "model.expert.up_proj.weight": torch.full(
                    (_INTERMEDIATE_DIM, _HIDDEN_DIM), 3, dtype=dtype
                ),
                "model.expert.down_proj.weight": torch.full(
                    (_HIDDEN_DIM, _INTERMEDIATE_DIM), 2, dtype=dtype
                ),
            }
        )
    )


def _adapter(source_dtype: torch.dtype, compute_dtype: torch.dtype) -> GgmlWeightAdapter:
    return GgmlWeightAdapter(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=source_dtype,
        compute_dtype=compute_dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ggml_adapter_builds_ready_cpu_views(dtype: torch.dtype) -> None:
    source = _source(dtype)
    adapter = _adapter(dtype, dtype)

    cpu_weight = adapter.make_cpu_weight(parse_safetensors(source))
    ready = adapter.make_ready_weight(cpu_weight)

    assert ready.dtype == dtype
    assert ready.gate_proj.data_ptr() == cpu_weight.gate_proj.data_ptr()
    assert adapter.cpu_extra_bytes() == 0
    assert adapter.ready_weight_bytes() > ready.storage_bytes


def test_ggml_adapter_converts_weights_once_during_ready_creation() -> None:
    adapter = _adapter(torch.float32, torch.bfloat16)

    cpu_weight = adapter.make_cpu_weight(parse_safetensors(_source(torch.float32)))
    ready = adapter.make_ready_weight(cpu_weight)

    assert cpu_weight.dtype == torch.float32
    assert ready.dtype == torch.bfloat16
    assert ready.gate_proj.data_ptr() != cpu_weight.gate_proj.data_ptr()


def test_ggml_adapter_rejects_wrong_source_dtype() -> None:
    parsed = parse_safetensors(_source(torch.float16))

    with pytest.raises(ValueError, match="unexpected dtype"):
        _adapter(torch.float32, torch.float32).make_cpu_weight(parsed)
