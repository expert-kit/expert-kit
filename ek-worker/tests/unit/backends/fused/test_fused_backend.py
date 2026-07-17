"""Numerical, ownership, and lifetime tests for the fused CUDA Backend."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import torch.nn.functional as functional

from expertkit_worker.backends import BackendBatch, InvalidBackendInput
from expertkit_worker.backends.fused import (
    FusedBackend,
    FusedCpuWeights,
    FusedExpertWeights,
    FusedWeightAdapter,
)
from expertkit_worker.weights import ReadyWeightTable

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_HIDDEN_DIM = 16
_INTERMEDIATE_DIM = 32
_TOP_K = 3


def _cpu_weight(seed: int, dtype: torch.dtype) -> FusedCpuWeights:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        value = torch.randn(rows, columns, generator=generator, dtype=torch.float32) / 4
        return value.to(dtype)

    return FusedCpuWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def _case(
    dtype: torch.dtype,
) -> tuple[
    BackendBatch,
    dict[int, FusedCpuWeights],
    ReadyWeightTable[FusedExpertWeights],
    FusedWeightAdapter,
]:
    device = torch.device("cuda:0")
    hidden = torch.linspace(
        -0.8,
        0.9,
        steps=4 * _HIDDEN_DIM,
        dtype=torch.float32,
    ).reshape(4, _HIDDEN_DIM)
    batch = BackendBatch(
        layer_id=1,
        hidden_states=hidden.to(device=device, dtype=dtype),
        expert_ids=torch.tensor(
            [[0, 2, -1], [1, 0, -1], [2, 1, 0], [1, 2, -1]],
            dtype=torch.int32,
            device=device,
        ),
        routing_weights=torch.tensor(
            [
                [0.35, 0.65, 0.0],
                [0.8, 0.2, 0.0],
                [0.15, 0.25, 0.6],
                [0.45, 0.55, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        distinct_expert_ids=(0, 1, 2),
    )
    source = {expert_id: _cpu_weight(17 + expert_id, dtype) for expert_id in range(3)}
    adapter = FusedWeightAdapter(
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=dtype,
        compute_dtype=dtype,
        device=device,
    )
    adapter.initialize_ready_storage(3)
    table: ReadyWeightTable[FusedExpertWeights] = ReadyWeightTable(2, 4)
    for expert_id, weight in source.items():
        table.publish(
            1,
            expert_id,
            adapter.make_ready_weight(weight, layer_id=1, expert_id=expert_id),
        )
    return batch, source, table, adapter


def _reference(
    batch: BackendBatch,
    weights: Mapping[int, FusedCpuWeights],
) -> torch.Tensor:
    hidden = batch.hidden_states.cpu().float()
    expert_ids = batch.expert_ids.cpu()
    routing = batch.routing_weights.cpu()
    output = torch.zeros_like(hidden)
    for token_index in range(batch.token_count):
        for route_index in range(batch.top_k):
            expert_id = int(expert_ids[token_index, route_index])
            if expert_id < 0:
                continue
            weight = weights[expert_id]
            token = hidden[token_index : token_index + 1]
            gate = functional.linear(token, weight.gate_proj.float())
            up = functional.linear(token, weight.up_proj.float())
            down = functional.linear(
                functional.silu(gate) * up,
                weight.down_proj.float(),
            )
            output[token_index] += down.squeeze(0) * routing[token_index, route_index]
    return output.to(batch.hidden_states.dtype)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float16, 6e-3, 6e-3),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_fused_backend_matches_reference_and_retains_slots_until_completion(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    batch, source, table, adapter = _case(dtype)
    backend = FusedBackend(
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=dtype,
        device="cuda:0",
        acquire_many=table.acquire_many,
    )
    output = torch.full_like(batch.hidden_states, torch.nan)
    output_pointer = output.data_ptr()

    completion = backend.submit(batch, output)
    completion.wait_host()

    assert output.data_ptr() == output_pointer
    torch.testing.assert_close(output.cpu(), _reference(batch, source), atol=atol, rtol=rtol)
    assert [table.usage_count(1, expert_id) for expert_id in range(3)] == [1, 1, 1]
    completion.close()
    completion.close()
    assert [table.usage_count(1, expert_id) for expert_id in range(3)] == [0, 0, 0]

    for expert_id in range(3):
        table.begin_withdrawal(1, expert_id)
        adapter.release_ready_weight(table.finish_withdrawal(1, expert_id))


def test_fused_backend_rejects_wrong_output_before_weight_lookup() -> None:
    batch, _source, table, adapter = _case(torch.float16)
    calls = 0

    def acquire(_layer_id: int, _expert_ids: tuple[int, ...]) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("weight lookup must not run")

    backend = FusedBackend(
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float16,
        device="cuda:0",
        acquire_many=acquire,  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidBackendInput, match="prepared output shape"):
        backend.submit(batch, torch.empty(2, _HIDDEN_DIM, device="cuda:0"))
    assert calls == 0

    for expert_id in range(3):
        table.begin_withdrawal(1, expert_id)
        adapter.release_ready_weight(table.finish_withdrawal(1, expert_id))


def test_fused_backend_resource_plan_covers_all_workspace_and_mapping_bytes() -> None:
    table: ReadyWeightTable[FusedExpertWeights] = ReadyWeightTable(2, 4)
    backend = FusedBackend(
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float16,
        device="cuda:0",
        acquire_many=table.acquire_many,
    )

    estimate = backend.estimate_resources(5)

    expected_workspace_elements = 5 * _TOP_K * (3 * _INTERMEDIATE_DIM + _HIDDEN_DIM)
    assert estimate.temporary_bytes_per_active_batch == expected_workspace_elements * 2
    assert estimate.shared_temporary_bytes == 2 * 4 * 4
