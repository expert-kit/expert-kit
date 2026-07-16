"""Tests for the shared Compute backend contract."""

from __future__ import annotations

import weakref

import pytest
import torch

from expertkit_worker.backends import (
    BackendBatch,
    BackendCapabilities,
    BackendFatalError,
    BackendFatalReason,
    BackendResourceEstimate,
    BackendWeightUnavailable,
    CompletedSubmission,
    ComputeBackend,
)


def make_batch() -> BackendBatch:
    """Return one valid CPU computation batch."""

    return BackendBatch(
        layer_id=2,
        hidden_states=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        expert_ids=torch.tensor([[1, -1], [3, 1], [3, -1]], dtype=torch.int32),
        routing_weights=torch.tensor(
            [[0.25, 0.0], [0.75, 0.25], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        distinct_expert_ids=(1, 3),
    )


def test_backend_batch_exposes_only_computation_fields() -> None:
    batch = make_batch()

    assert batch.token_count == 3
    assert batch.hidden_dim == 4
    assert batch.top_k == 2
    assert not hasattr(batch, "instance_id")
    assert not hasattr(batch, "topology_version")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("hidden_states", torch.ones(3, 4, dtype=torch.int32), "FP16, BF16, or FP32"),
        ("expert_ids", torch.ones(3, 2, dtype=torch.int64), "must use int32"),
        ("routing_weights", torch.ones(3, 2, dtype=torch.float16), "must use FP32"),
        ("distinct_expert_ids", (3, 1), "sorted with no duplicates"),
    ],
)
def test_backend_batch_rejects_invalid_metadata(
    field: str,
    value: object,
    match: str,
) -> None:
    fields = {
        "layer_id": 2,
        "hidden_states": torch.ones(3, 4),
        "expert_ids": torch.ones(3, 2, dtype=torch.int32),
        "routing_weights": torch.ones(3, 2, dtype=torch.float32),
        "distinct_expert_ids": (1, 3),
    }
    fields[field] = value

    with pytest.raises(ValueError, match=match):
        BackendBatch(**fields)  # type: ignore[arg-type]


def test_backend_batch_rejects_noncontiguous_common_inputs() -> None:
    batch = make_batch()

    with pytest.raises(ValueError, match="hidden_states must be contiguous"):
        BackendBatch(
            layer_id=batch.layer_id,
            hidden_states=torch.ones(3, 8)[:, ::2],
            expert_ids=batch.expert_ids,
            routing_weights=batch.routing_weights,
            distinct_expert_ids=batch.distinct_expert_ids,
        )


def test_dynamic_capabilities_validate_worker_limits() -> None:
    capabilities = BackendCapabilities(
        supports_dynamic_tokens=True,
        supports_concurrent_batches=False,
        required_assignment_alignment=8,
        max_batch_tokens=64,
        max_assignments_per_batch=128,
        supported_token_profiles=(16, 64),
    )

    capabilities.validate_runtime(max_batch_tokens=64, active_batches=1)
    with pytest.raises(ValueError, match="max_batch_tokens exceeds"):
        capabilities.validate_runtime(max_batch_tokens=65, active_batches=1)
    with pytest.raises(ValueError, match="does not support concurrent"):
        capabilities.validate_runtime(max_batch_tokens=64, active_batches=2)


def test_fixed_token_capabilities_require_profiles() -> None:
    with pytest.raises(ValueError, match="must declare supported_token_profiles"):
        BackendCapabilities(
            supports_dynamic_tokens=False,
            supports_concurrent_batches=True,
        )

    capabilities = BackendCapabilities(
        supports_dynamic_tokens=False,
        supports_concurrent_batches=True,
        supported_token_profiles=(8, 32),
    )
    capabilities.validate_runtime(max_batch_tokens=32, active_batches=2)
    with pytest.raises(ValueError, match="largest Backend token profile"):
        capabilities.validate_runtime(max_batch_tokens=33, active_batches=2)


def test_resource_estimate_multiplies_only_per_active_memory() -> None:
    estimate = BackendResourceEstimate(
        temporary_bytes_per_active_batch=1_024,
        shared_temporary_bytes=256,
    )

    assert estimate.total_bytes(3) == 3_328


def test_completed_submission_retains_resources_until_close() -> None:
    class Resource:
        pass

    resource = Resource()
    reference = weakref.ref(resource)
    completion = CompletedSubmission((resource,))
    del resource

    completion.wait_host()
    assert reference() is not None
    completion.close()
    completion.close()
    assert reference() is None


def test_weight_unavailable_preserves_expert_ids() -> None:
    error = BackendWeightUnavailable((2, 7))

    assert error.unavailable_expert_ids == (2, 7)


def test_fatal_error_uses_structured_reason() -> None:
    error = BackendFatalError(BackendFatalReason.DEVICE_OOM, "allocation failed")

    assert error.reason is BackendFatalReason.DEVICE_OOM
    assert error.diagnostic == "allocation failed"


def test_compute_backend_requires_every_contract_method() -> None:
    class IncompleteBackend(ComputeBackend):
        pass

    with pytest.raises(TypeError, match="abstract class"):
        IncompleteBackend()
