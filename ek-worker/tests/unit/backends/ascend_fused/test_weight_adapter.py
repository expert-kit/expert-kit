# ruff: noqa: E402

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save

sys.modules.setdefault("torch_npu", ModuleType("torch_npu"))

from expertkit_worker.backends.ascend_fused.weights import (
    AscendFusedCpuWeights,
    AscendFusedModelSlimW8A8CpuWeights,
    AscendFusedModelSlimW8A8ExpertWeights,
    AscendFusedModelSlimW8A8WeightAdapter,
    AscendFusedWeightAdapter,
)
from expertkit_worker.backends.factory import create_compute_backend, create_weight_adapter
from expertkit_worker.config import BackendName, QuantizationType
from expertkit_worker.weights import parse_safetensors


class _FakeNpuRuntime:
    device = SimpleNamespace(type="npu", index=0)

    def capture_current_work(self):
        return SimpleNamespace(wait_host=lambda: None)


def _values(prefix: str = "layers.0.ffn.experts.0") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for index, (name, shape) in enumerate(
        (("w1", (8, 4)), ("w3", (8, 4)), ("w2", (4, 8))),
        start=1,
    ):
        matrix = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int8).reshape(shape)
        result[f"{prefix}.{name}.weight"] = matrix + index
        result[f"{prefix}.{name}.weight_scale"] = torch.full(
            (shape[0], 1),
            index / 10,
            dtype=torch.float32,
        )
        result[f"{prefix}.{name}.weight_offset"] = torch.zeros(
            shape[0],
            1,
            dtype=torch.float32,
        )
    return result


def _adapter(
    compute_dtype: torch.dtype = torch.bfloat16,
) -> AscendFusedModelSlimW8A8WeightAdapter:
    return AscendFusedModelSlimW8A8WeightAdapter(
        hidden_dim=4,
        intermediate_dim=8,
        runtime=_FakeNpuRuntime(),  # type: ignore[arg-type]
        compute_dtype=compute_dtype,
    )


def _float_adapter() -> AscendFusedWeightAdapter:
    return AscendFusedWeightAdapter(
        num_layers=1,
        experts_per_layer=4,
        hidden_dim=4,
        intermediate_dim=8,
        source_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        runtime=_FakeNpuRuntime(),  # type: ignore[arg-type]
    )


def test_modelslim_adapter_packs_gmm_weights_and_scales() -> None:
    values = _values()
    cpu = _adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))

    ready = cpu.copy(torch.device("cpu"), torch.bfloat16)

    torch.testing.assert_close(ready.w13[:, :8], values["layers.0.ffn.experts.0.w1.weight"].T)
    torch.testing.assert_close(ready.w13[:, 8:], values["layers.0.ffn.experts.0.w3.weight"].T)
    torch.testing.assert_close(ready.w2, values["layers.0.ffn.experts.0.w2.weight"].T)
    torch.testing.assert_close(
        ready.w13_scale,
        torch.cat(
            (
                values["layers.0.ffn.experts.0.w1.weight_scale"].flatten(),
                values["layers.0.ffn.experts.0.w3.weight_scale"].flatten(),
            )
        ).to(torch.bfloat16),
    )
    torch.testing.assert_close(
        ready.w2_scale,
        values["layers.0.ffn.experts.0.w2.weight_scale"].flatten().to(torch.bfloat16),
    )
    assert ready.dtype is torch.bfloat16
    assert ready.w13_scale.dtype is torch.bfloat16
    assert ready.w2_scale.dtype is torch.bfloat16
    assert ready.hidden_dim == 4
    assert ready.intermediate_dim == 8
    assert ready.storage_bytes == _adapter().ready_weight_bytes()


def test_modelslim_adapter_accepts_projection_names() -> None:
    values = {
        name.replace(".w1.", ".gate_proj.")
        .replace(".w3.", ".up_proj.")
        .replace(".w2.", ".down_proj."): value
        for name, value in _values().items()
    }

    cpu = _adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))

    assert cpu.matrices[0].shape == (8, 4)


def test_modelslim_adapter_accounts_for_source_and_ready_bytes() -> None:
    bf16_adapter = _adapter(torch.bfloat16)
    fp16_adapter = _adapter(torch.float16)

    assert bf16_adapter.cpu_extra_bytes() == 0
    assert bf16_adapter.conversion_temporary_bytes() == 0
    assert bf16_adapter.source_tensor_bytes() == 3 * 4 * 8 + 2 * (2 * 8 + 4) * 4
    assert bf16_adapter.ready_weight_bytes() == 3 * 4 * 8 + (2 * 8 + 4) * 2
    assert fp16_adapter.ready_weight_bytes() == 3 * 4 * 8 + (2 * 8 + 4) * 4


def test_modelslim_adapter_uses_fp32_scales_for_fp16_output() -> None:
    values = _values()
    cpu = _adapter(torch.float16).make_cpu_weight(
        parse_safetensors(bytearray(save(values)))
    )

    ready = cpu.copy(torch.device("cpu"), torch.float16)

    assert ready.dtype is torch.float16
    assert ready.w13_scale.dtype is torch.float32
    assert ready.w2_scale.dtype is torch.float32


def test_float_adapter_rejects_cached_weight_from_another_model() -> None:
    cached = AscendFusedCpuWeights(
        gate_proj=torch.empty((7, 4), dtype=torch.bfloat16),
        up_proj=torch.empty((7, 4), dtype=torch.bfloat16),
        down_proj=torch.empty((4, 7), dtype=torch.bfloat16),
    )

    with pytest.raises(ValueError, match="shape does not match"):
        _float_adapter().make_ready_weight(cached, layer_id=0, expert_id=0)


def test_float_adapter_rejects_cached_weight_with_wrong_source_dtype() -> None:
    cached = AscendFusedCpuWeights(
        gate_proj=torch.empty((8, 4), dtype=torch.float32),
        up_proj=torch.empty((8, 4), dtype=torch.float32),
        down_proj=torch.empty((4, 8), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="dtype does not match"):
        _float_adapter().make_ready_weight(cached, layer_id=0, expert_id=0)


def test_modelslim_adapter_rejects_cached_weight_from_another_model() -> None:
    cached = AscendFusedModelSlimW8A8CpuWeights(
        matrices=(
            torch.empty((7, 4), dtype=torch.int8),
            torch.empty((7, 4), dtype=torch.int8),
            torch.empty((4, 7), dtype=torch.int8),
        ),
        scales=(
            torch.ones((7, 1)),
            torch.ones((7, 1)),
            torch.ones((4, 1)),
        ),
    )

    with pytest.raises(ValueError, match="shape does not match"):
        _adapter().make_ready_weight(cached, layer_id=0, expert_id=0)


@pytest.mark.parametrize("field", ["weight_scale", "weight_offset"])
def test_modelslim_adapter_rejects_missing_auxiliary(field: str) -> None:
    values = _values()
    del values[f"layers.0.ffn.experts.0.w1.{field}"]

    with pytest.raises(ValueError, match="ModelSlim"):
        _adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))


def test_modelslim_adapter_rejects_nonzero_offsets() -> None:
    values = _values()
    values["layers.0.ffn.experts.0.w1.weight_offset"][0] = 1

    with pytest.raises(ValueError, match="nonzero"):
        _adapter().make_cpu_weight(parse_safetensors(bytearray(save(values))))


def test_modelslim_ready_weight_rejects_bad_scale_shape() -> None:
    with pytest.raises(ValueError, match="scale shapes"):
        AscendFusedModelSlimW8A8ExpertWeights(
            w13=torch.ones((4, 16), dtype=torch.int8),
            w2=torch.ones((8, 4), dtype=torch.int8),
            w13_scale=torch.ones(8),
            w2_scale=torch.ones(4),
            dtype=torch.bfloat16,
        )


def test_modelslim_ready_weight_rejects_scale_dtype_incompatible_with_output() -> None:
    with pytest.raises(ValueError, match="scale dtype"):
        AscendFusedModelSlimW8A8ExpertWeights(
            w13=torch.ones((4, 16), dtype=torch.int8),
            w2=torch.ones((8, 4), dtype=torch.int8),
            w13_scale=torch.ones(16),
            w2_scale=torch.ones(4),
            dtype=torch.bfloat16,
        )


def test_factory_selects_modelslim_adapter_and_dynamic_compute() -> None:
    config = SimpleNamespace(
        worker=SimpleNamespace(backend=BackendName.ASCEND_FUSED),
        model=SimpleNamespace(
            num_layers=1,
            experts_per_layer=4,
            hidden_dim=4,
            expert_intermediate_dim=8,
            top_k=2,
            swiglu_limit=5.0,
            quantization=SimpleNamespace(type=QuantizationType.MODELSLIM_W8A8_DYNAMIC),
        ),
    )
    runtime = _FakeNpuRuntime()

    adapter = create_weight_adapter(
        config,  # type: ignore[arg-type]
        source_dtype=torch.bfloat16,
        compute_dtype=torch.bfloat16,
        runtime=runtime,  # type: ignore[arg-type]
    )
    backend = create_compute_backend(
        config,  # type: ignore[arg-type]
        dtype=torch.bfloat16,
        runtime=runtime,  # type: ignore[arg-type]
        acquire_many=lambda _layer, _experts: None,
    )

    assert isinstance(adapter, AscendFusedModelSlimW8A8WeightAdapter)
    assert backend.capabilities.supports_concurrent_batches is True
    assert backend._compute_type.value == "dynamic_w8a8"  # type: ignore[attr-defined]
    assert backend._swiglu_limit == 5.0  # type: ignore[attr-defined]
