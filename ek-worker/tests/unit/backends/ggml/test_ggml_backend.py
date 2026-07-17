"""Numerical and lifetime tests for the experimental GGML Backend."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

try:
    version("ggml-python")
except PackageNotFoundError:
    pytest.skip("the GGML extra is not installed", allow_module_level=True)

from expertkit_worker.backends import BackendBatch, BackendWeightUnavailable, InvalidBackendInput
from expertkit_worker.backends.ggml import GgmlBackend, GgmlExpertWeights
from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights
from expertkit_worker.weights import ReadyWeightTable

_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 6
_TOP_K = 3


def _torch_weight(seed: int, dtype: torch.dtype) -> TorchExpertWeights:
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return (torch.randn(rows, columns, generator=generator) / 4).to(dtype)

    return TorchExpertWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def _batch(dtype: torch.dtype) -> BackendBatch:
    return BackendBatch(
        layer_id=0,
        hidden_states=torch.tensor(
            [
                [0.2, -0.4, 0.8, 0.5],
                [-0.1, 0.7, 0.3, -0.6],
                [0.9, 0.1, -0.2, 0.4],
            ],
            dtype=dtype,
        ),
        expert_ids=torch.tensor(
            [[0, 2, -1], [1, 0, -1], [2, 1, 0]],
            dtype=torch.int32,
        ),
        routing_weights=torch.tensor(
            [[0.35, 0.65, 0.0], [0.8, 0.2, 0.0], [0.15, 0.25, 0.6]],
            dtype=torch.float32,
        ),
        distinct_expert_ids=(0, 1, 2),
    )


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 3e-3, 3e-3),
        (torch.bfloat16, 5e-2, 2e-2),
    ],
)
def test_ggml_backend_matches_torch_for_weighted_worker_batch(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    batch = _batch(dtype)
    torch_table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    ggml_table: ReadyWeightTable[GgmlExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id in range(3):
        source = _torch_weight(17 + expert_id, dtype)
        torch_table.publish(0, expert_id, source)
        ggml_table.publish(
            0,
            expert_id,
            GgmlExpertWeights(
                gate_proj=source.gate_proj,
                up_proj=source.up_proj,
                down_proj=source.down_proj,
            ),
        )

    torch_backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=dtype,
        device="cpu",
        acquire_many=torch_table.acquire_many,
    )
    ggml_backend = GgmlBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=dtype,
        cpu_threads=2,
        acquire_many=ggml_table.acquire_many,
    )
    torch_output = torch.empty_like(batch.hidden_states)
    ggml_output = torch.empty_like(batch.hidden_states)

    torch_completion = torch_backend.submit(batch, torch_output)
    ggml_completion = ggml_backend.submit(batch, ggml_output)
    torch_completion.wait_host()
    ggml_completion.wait_host()

    torch.testing.assert_close(ggml_output, torch_output, atol=atol, rtol=rtol)
    assert [ggml_table.usage_count(0, expert_id) for expert_id in range(3)] == [1, 1, 1]
    torch_completion.close()
    ggml_completion.close()
    ggml_completion.close()
    assert [ggml_table.usage_count(0, expert_id) for expert_id in range(3)] == [0, 0, 0]


def test_ggml_backend_reports_missing_weight_without_partial_retention() -> None:
    batch = _batch(torch.float32)
    table: ReadyWeightTable[GgmlExpertWeights] = ReadyWeightTable(1, 3)
    source = _torch_weight(1, torch.float32)
    table.publish(
        0,
        0,
        GgmlExpertWeights(
            gate_proj=source.gate_proj,
            up_proj=source.up_proj,
            down_proj=source.down_proj,
        ),
    )
    backend = GgmlBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        cpu_threads=1,
        acquire_many=table.acquire_many,
    )

    with pytest.raises(BackendWeightUnavailable) as caught:
        backend.submit(batch, torch.empty_like(batch.hidden_states))

    assert caught.value.unavailable_expert_ids == (1, 2)
    assert table.usage_count(0, 0) == 0


def test_ggml_backend_rejects_cuda_boundary_before_weight_lookup() -> None:
    batch = _batch(torch.float32)
    calls = 0

    def acquire(_layer_id: int, _expert_ids: tuple[int, ...]) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("acquisition must not run")

    backend = GgmlBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        cpu_threads=1,
        acquire_many=acquire,  # type: ignore[arg-type]
    )
    wrong_output = torch.empty_like(batch.hidden_states, device="meta")

    with pytest.raises(InvalidBackendInput, match="CPU output"):
        backend.submit(batch, wrong_output)
    assert calls == 0


def test_ggml_backend_resource_estimate_scales_with_batch_limit() -> None:
    table: ReadyWeightTable[GgmlExpertWeights] = ReadyWeightTable(1, 3)
    backend = GgmlBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float16,
        cpu_threads=1,
        acquire_many=table.acquire_many,
    )

    one = backend.estimate_resources(1).temporary_bytes_per_active_batch
    eight = backend.estimate_resources(8).temporary_bytes_per_active_batch

    assert one > 0
    assert eight > one
