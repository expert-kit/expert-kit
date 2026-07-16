"""Tests for Transport-independent Worker batch validation."""

import pytest
import torch

from expertkit_transport.contracts import RoutedLayerBatch, WorkerBatch


def make_batch(**changes: object) -> WorkerBatch:
    values: dict[str, object] = {
        "instance_id": 7,
        "layer_id": 2,
        "topology_version": 11,
        "hidden_states": torch.zeros((4, 8), dtype=torch.bfloat16),
        "token_indices": torch.tensor([0, 2], dtype=torch.int64),
        "expert_ids": torch.tensor([[3, -1], [5, 3]], dtype=torch.int32),
        "routing_weights": torch.tensor([[0.5, 0.0], [0.25, 0.75]], dtype=torch.float32),
        "distinct_expert_ids": (3, 5),
    }
    values.update(changes)
    return WorkerBatch(**values)  # type: ignore[arg-type]


def test_selected_rows_define_worker_batch_shape() -> None:
    batch = make_batch()

    assert batch.token_count == 2
    assert batch.hidden_dim == 8
    assert batch.top_k == 2
    assert batch.hidden_states.shape == (4, 8)


def test_omitted_indices_select_every_source_row() -> None:
    batch = make_batch(
        token_indices=None,
        expert_ids=torch.full((4, 2), -1, dtype=torch.int32),
        routing_weights=torch.zeros((4, 2), dtype=torch.float32),
        distinct_expert_ids=(),
    )

    assert batch.token_count == 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"instance_id": -1}, "instance_id"),
        ({"hidden_states": torch.tensor(1.0)}, "source_tokens"),
        ({"hidden_states": torch.zeros((2, 8), dtype=torch.int8)}, "FP16, BF16, or FP32"),
        ({"token_indices": torch.tensor([0, 1], dtype=torch.int32)}, "int64"),
        ({"expert_ids": torch.zeros((2, 2), dtype=torch.int64)}, "int32"),
        ({"routing_weights": torch.zeros((2, 2), dtype=torch.float16)}, "FP32"),
        ({"distinct_expert_ids": (5, 3)}, "sorted"),
        ({"distinct_expert_ids": (3, 3)}, "duplicates"),
    ],
)
def test_batch_rejects_invalid_metadata(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_batch(**change)


def test_batch_validation_does_not_read_tensor_values() -> None:
    batch = make_batch(
        hidden_states=torch.empty((4, 8), dtype=torch.float16, device="meta"),
        token_indices=torch.empty((2,), dtype=torch.int64, device="meta"),
        expert_ids=torch.empty((2, 2), dtype=torch.int32, device="meta"),
        routing_weights=torch.empty((2, 2), dtype=torch.float32, device="meta"),
    )

    assert batch.token_count == 2


def test_batch_copies_distinct_expert_sequence() -> None:
    distinct = [3, 5]
    batch = make_batch(distinct_expert_ids=distinct)

    distinct.append(7)

    assert batch.distinct_expert_ids == (3, 5)


def test_routed_layer_requires_fixed_external_dtypes() -> None:
    with pytest.raises(ValueError, match="int32"):
        RoutedLayerBatch(
            instance_id=7,
            layer_id=2,
            hidden_states=torch.zeros((2, 8), dtype=torch.bfloat16),
            expert_ids=torch.zeros((2, 2), dtype=torch.int64),
            routing_weights=torch.zeros((2, 2), dtype=torch.float32),
        )
