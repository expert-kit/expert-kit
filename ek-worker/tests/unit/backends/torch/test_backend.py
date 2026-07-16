"""Numerical and lifetime tests for the Torch eager Backend."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
import torch.nn.functional as functional

from expertkit_worker.backends import (
    BackendBatch,
    BackendFatalError,
    BackendFatalReason,
    BackendWeightUnavailable,
    InvalidBackendInput,
)
from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights
from expertkit_worker.weights import ReadyWeightTable

_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 6
_TOP_K = 3


def make_weight(seed: int, dtype: torch.dtype, device: torch.device) -> TorchExpertWeights:
    """Return a deterministic small expert fixture."""

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        value = torch.randn(rows, columns, generator=generator, dtype=torch.float32) / 4
        return value.to(device=device, dtype=dtype)

    return TorchExpertWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def make_case(
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[BackendBatch, dict[int, TorchExpertWeights]]:
    """Return routing with multiple experts, duplicate Tokens, and invalid slots."""

    hidden = torch.tensor(
        [
            [0.2, -0.4, 0.8, 0.5],
            [-0.1, 0.7, 0.3, -0.6],
            [0.9, 0.1, -0.2, 0.4],
        ],
        dtype=dtype,
        device=device,
    )
    expert_ids = torch.tensor(
        [[0, 2, -1], [1, 0, -1], [2, 1, 0]],
        dtype=torch.int32,
        device=device,
    )
    routing = torch.tensor(
        [[0.35, 0.65, 0.0], [0.8, 0.2, 0.0], [0.15, 0.25, 0.6]],
        dtype=torch.float32,
        device=device,
    )
    batch = BackendBatch(
        layer_id=0,
        hidden_states=hidden,
        expert_ids=expert_ids,
        routing_weights=routing,
        distinct_expert_ids=(0, 1, 2),
    )
    return batch, {expert_id: make_weight(17 + expert_id, dtype, device) for expert_id in range(3)}


def reference_output(
    batch: BackendBatch,
    weights: Mapping[int, TorchExpertWeights],
) -> torch.Tensor:
    """Compute an independent assignment-by-assignment FP32 reference."""

    result = torch.zeros(batch.token_count, batch.hidden_dim, dtype=torch.float32)
    host_hidden = batch.hidden_states.detach().cpu()
    host_ids = batch.expert_ids.detach().cpu()
    host_routing = batch.routing_weights.detach().cpu()
    host_weights = {
        expert_id: tuple(tensor.detach().cpu() for tensor in weight.tensors)
        for expert_id, weight in weights.items()
    }
    for token_index in range(batch.token_count):
        for route_index in range(batch.top_k):
            expert_id = int(host_ids[token_index, route_index])
            if expert_id < 0:
                continue
            gate_weight, up_weight, down_weight = host_weights[expert_id]
            hidden = host_hidden[token_index : token_index + 1]
            gate = hidden @ gate_weight.transpose(0, 1)
            up = hidden @ up_weight.transpose(0, 1)
            output = (functional.silu(gate) * up) @ down_weight.transpose(0, 1)
            result[token_index] += (
                output.float().squeeze(0) * host_routing[token_index, route_index]
            )
    return result.to(batch.hidden_states.dtype)


def make_backend(
    table: ReadyWeightTable[TorchExpertWeights],
    dtype: torch.dtype,
    device: torch.device,
) -> TorchBackend:
    """Build a Backend over the ready-weight table fixture."""

    return TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=dtype,
        device=device,
        acquire_many=table.acquire_many,
    )


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-6, 1e-5),
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_torch_backend_matches_weighted_reference_and_releases_weights(
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    device = torch.device("cpu")
    batch, weights = make_case(dtype, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, dtype, device)
    output = torch.full_like(batch.hidden_states, torch.nan)

    completion = backend.submit(batch, output)

    completion.wait_host()
    torch.testing.assert_close(output, reference_output(batch, weights), atol=atol, rtol=rtol)
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [1, 1, 1]
    completion.close()
    completion.close()
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [0, 0, 0]


def test_torch_backend_acquires_all_experts_once() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    calls: list[tuple[int, tuple[int, ...]]] = []

    def acquire(layer_id: int, expert_ids: tuple[int, ...]):
        calls.append((layer_id, expert_ids))
        return table.acquire_many(layer_id, expert_ids)

    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        device=device,
        acquire_many=acquire,
    )
    completion = backend.submit(batch, torch.empty_like(batch.hidden_states))
    completion.wait_host()
    completion.close()

    assert calls == [(0, (0, 1, 2))]


def test_torch_backend_reports_missing_ready_weights_without_partial_retention() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    table.publish(0, 0, weights[0])
    table.publish(0, 2, weights[2])
    backend = make_backend(table, torch.float32, device)

    with pytest.raises(BackendWeightUnavailable) as caught:
        backend.submit(batch, torch.empty_like(batch.hidden_states))

    assert caught.value.unavailable_expert_ids == (1,)
    assert table.usage_count(0, 0) == 0
    assert table.usage_count(0, 2) == 0


def test_torch_backend_treats_invalid_ready_weight_as_fatal_and_releases_lease() -> None:
    device = torch.device("cpu")
    batch, weights = make_case(torch.float32, device)
    weights[0] = TorchExpertWeights(
        gate_proj=torch.ones(_INTERMEDIATE_DIM, _HIDDEN_DIM - 1),
        up_proj=torch.ones(_INTERMEDIATE_DIM, _HIDDEN_DIM - 1),
        down_proj=torch.ones(_HIDDEN_DIM - 1, _INTERMEDIATE_DIM),
    )
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, torch.float32, device)

    with pytest.raises(BackendFatalError) as caught:
        backend.submit(batch, torch.empty_like(batch.hidden_states))

    assert caught.value.reason is BackendFatalReason.UNEXPECTED
    assert [table.usage_count(0, expert_id) for expert_id in range(3)] == [0, 0, 0]


def test_torch_backend_zeroes_output_when_worker_has_no_local_assignments() -> None:
    device = torch.device("cpu")
    hidden = torch.ones(2, _HIDDEN_DIM)
    batch = BackendBatch(
        layer_id=0,
        hidden_states=hidden,
        expert_ids=torch.full((2, _TOP_K), -1, dtype=torch.int32),
        routing_weights=torch.zeros(2, _TOP_K, dtype=torch.float32),
        distinct_expert_ids=(),
    )
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    backend = make_backend(table, torch.float32, device)
    output = torch.full_like(hidden, 7)

    completion = backend.submit(batch, output)
    completion.wait_host()
    completion.close()

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_torch_backend_rejects_wrong_prepared_output_before_weight_acquisition() -> None:
    device = torch.device("cpu")
    batch, _ = make_case(torch.float32, device)
    calls = 0

    def acquire(
        _layer_id: int,
        _expert_ids: tuple[int, ...],
    ) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("acquisition must not run")

    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        device=device,
        acquire_many=acquire,  # type: ignore[arg-type]
    )

    with pytest.raises(InvalidBackendInput, match="prepared output shape"):
        backend.submit(batch, torch.empty(2, _HIDDEN_DIM))
    assert calls == 0


def test_torch_backend_resource_estimate_scales_with_token_bound() -> None:
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    backend = make_backend(table, torch.float16, torch.device("cpu"))

    one = backend.estimate_resources(1).temporary_bytes_per_active_batch
    eight = backend.estimate_resources(8).temporary_bytes_per_active_batch

    assert one > 0
    assert eight == one * 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_torch_backend_completes_repeated_cuda_submissions() -> None:
    device = torch.device("cuda:0")
    batch, weights = make_case(torch.float16, device)
    table: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    for expert_id, weight in weights.items():
        table.publish(0, expert_id, weight)
    backend = make_backend(table, torch.float16, device)
    expected = reference_output(batch, weights)

    for _ in range(3):
        output = torch.empty_like(batch.hidden_states)
        completion = backend.submit(batch, output)
        completion.wait_host()
        completion.close()
        torch.testing.assert_close(output.cpu(), expected.cpu(), atol=2e-3, rtol=2e-3)
