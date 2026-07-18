"""Tests for Worker grouping and physical batch splitting."""

import pytest
import torch

from expertkit_transport.contracts import (
    OutputBufferProvider,
    PreparedOutput,
    RoutedLayerBatch,
    TransportError,
    TransportErrorCode,
    WorkerBatch,
    WorkerTransport,
)
from expertkit_transport.orchestration import (
    RoundRobinSelector,
    TopologySnapshot,
    WorkerIdentity,
    WorkerTarget,
    group_worker_batches,
)


class FakeTransport(WorkerTransport):
    @property
    def output_buffers(self) -> OutputBufferProvider:
        raise NotImplementedError

    async def start(self) -> None:
        return None

    async def submit(
        self,
        batch: WorkerBatch,
        output: PreparedOutput,
        *,
        monotonic_deadline: float,
    ) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def target(name: str, *, max_batch_tokens: int = 8) -> WorkerTarget:
    return WorkerTarget(
        identity=WorkerIdentity(worker_id=name, start_id=f"{name}-start"),
        transport=FakeTransport(),
        max_batch_tokens=max_batch_tokens,
        max_active_batches=1,
        max_pending_batches=1,
    )


def routed_batch() -> RoutedLayerBatch:
    return RoutedLayerBatch(
        instance_id=7,
        layer_id=2,
        hidden_states=torch.arange(24, dtype=torch.float32).reshape(3, 8),
        expert_ids=torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int32),
        routing_weights=torch.tensor(
            [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]],
            dtype=torch.float32,
        ),
        distinct_expert_ids=(0, 1, 2),
    )


def topology(worker_a: WorkerTarget, worker_b: WorkerTarget) -> TopologySnapshot:
    return TopologySnapshot(
        instance_id=7,
        version=11,
        routes={
            (2, 0): (worker_a,),
            (2, 1): (worker_b,),
            (2, 2): (worker_a,),
        },
    )


def plans_by_worker(plans: tuple) -> dict[str, WorkerBatch]:
    return {plan.target.identity.worker_id: plan.batch for plan in plans}


def test_grouping_keeps_one_row_and_fixed_top_k_per_worker() -> None:
    worker_a = target("worker-a")
    worker_b = target("worker-b")
    source = routed_batch()

    grouped = plans_by_worker(
        group_worker_batches(source, topology(worker_a, worker_b), RoundRobinSelector())
    )

    a = grouped["worker-a"]
    assert a.hidden_states is source.hidden_states
    assert a.token_indices.tolist() == [0, 1, 2]
    assert a.expert_ids.tolist() == [[0, -1], [-1, 2], [2, 0]]
    torch.testing.assert_close(
        a.routing_weights,
        torch.tensor([[0.7, 0.0], [0.0, 0.8], [0.6, 0.4]], dtype=torch.float32),
    )
    assert a.distinct_expert_ids == (0, 2)

    b = grouped["worker-b"]
    assert b.hidden_states is source.hidden_states
    assert b.token_indices.tolist() == [0, 1]
    assert b.expert_ids.tolist() == [[-1, 1], [1, -1]]
    torch.testing.assert_close(
        b.routing_weights,
        torch.tensor([[0.0, 0.3], [0.2, 0.0]], dtype=torch.float32),
    )
    assert b.distinct_expert_ids == (1,)


def test_single_worker_reuses_complete_routing_tensors() -> None:
    worker = target("worker-a")
    source = routed_batch()
    snapshot = TopologySnapshot(
        instance_id=7,
        version=11,
        routes={(2, expert_id): (worker,) for expert_id in (0, 1, 2)},
    )

    plans = group_worker_batches(source, snapshot, RoundRobinSelector())

    assert len(plans) == 1
    physical = plans[0].batch
    assert physical.hidden_states is source.hidden_states
    assert physical.expert_ids is source.expert_ids
    assert physical.routing_weights is source.routing_weights
    assert physical.token_indices is None
    assert physical.distinct_expert_ids == (0, 1, 2)


def test_grouping_splits_only_between_token_rows() -> None:
    worker_a = target("worker-a", max_batch_tokens=1)
    worker_b = target("worker-b", max_batch_tokens=2)

    plans = group_worker_batches(
        routed_batch(),
        topology(worker_a, worker_b),
        RoundRobinSelector(),
    )

    a_plans = [plan.batch for plan in plans if plan.target.identity == worker_a.identity]
    b_plans = [plan.batch for plan in plans if plan.target.identity == worker_b.identity]
    assert [batch.token_indices.tolist() for batch in a_plans] == [[0], [1], [2]]
    assert [batch.token_count for batch in a_plans] == [1, 1, 1]
    assert [batch.token_indices.tolist() for batch in b_plans] == [[0, 1]]
    assert a_plans[2].expert_ids.tolist() == [[2, 0]]
    assert [batch.distinct_expert_ids for batch in a_plans] == [
        (0,),
        (2,),
        (0, 2),
    ]
    assert b_plans[0].distinct_expert_ids == (1,)


def test_single_worker_split_keeps_exact_chunk_expert_lists() -> None:
    worker = target("worker-a", max_batch_tokens=2)
    source = routed_batch()
    snapshot = TopologySnapshot(
        instance_id=7,
        version=11,
        routes={(2, expert_id): (worker,) for expert_id in (0, 1, 2)},
    )

    plans = group_worker_batches(source, snapshot, RoundRobinSelector())

    assert [plan.batch.token_indices.tolist() for plan in plans] == [[0, 1], [2]]
    assert [plan.batch.distinct_expert_ids for plan in plans] == [(0, 1, 2), (0, 2)]


def test_round_robin_selects_one_replica_per_expert_per_call() -> None:
    worker_a = target("worker-a")
    worker_b = target("worker-b")
    snapshot = TopologySnapshot(
        instance_id=7,
        version=11,
        routes={(2, 0): (worker_a, worker_b)},
    )
    batch = RoutedLayerBatch(
        instance_id=7,
        layer_id=2,
        hidden_states=torch.zeros((2, 4), dtype=torch.float16),
        expert_ids=torch.zeros((2, 1), dtype=torch.int32),
        routing_weights=torch.ones((2, 1), dtype=torch.float32),
        distinct_expert_ids=(0,),
    )
    selector = RoundRobinSelector()

    first = group_worker_batches(batch, snapshot, selector)
    second = group_worker_batches(batch, snapshot, selector)

    assert len(first) == len(second) == 1
    assert first[0].target.identity == worker_a.identity
    assert second[0].target.identity == worker_b.identity


def test_grouping_reports_every_expert_without_a_ready_route() -> None:
    worker_a = target("worker-a")
    snapshot = TopologySnapshot(
        instance_id=7,
        version=11,
        routes={(2, 0): (worker_a,)},
    )

    with pytest.raises(TransportError) as caught:
        group_worker_batches(routed_batch(), snapshot, RoundRobinSelector())

    assert caught.value.code is TransportErrorCode.UNAVAILABLE
    assert caught.value.retryable is True
    assert caught.value.unavailable_expert_ids == (1, 2)


def test_retry_selection_can_exclude_failed_worker_start() -> None:
    worker_a = target("worker-a")
    worker_b = target("worker-b")
    snapshot = TopologySnapshot(
        instance_id=7,
        version=11,
        routes={(2, 0): (worker_a, worker_b)},
    )
    batch = RoutedLayerBatch(
        instance_id=7,
        layer_id=2,
        hidden_states=torch.zeros((1, 4), dtype=torch.float16),
        expert_ids=torch.zeros((1, 1), dtype=torch.int32),
        routing_weights=torch.ones((1, 1), dtype=torch.float32),
        distinct_expert_ids=(0,),
    )

    plans = group_worker_batches(
        batch,
        snapshot,
        RoundRobinSelector(),
        excluded=frozenset((worker_a.identity,)),
    )

    assert plans[0].target.identity == worker_b.identity


def test_retry_selection_fails_when_every_replica_is_excluded() -> None:
    worker_a = target("worker-a")
    snapshot = TopologySnapshot(
        instance_id=7,
        version=11,
        routes={(2, 0): (worker_a,)},
    )
    batch = RoutedLayerBatch(
        instance_id=7,
        layer_id=2,
        hidden_states=torch.zeros((1, 4), dtype=torch.float16),
        expert_ids=torch.zeros((1, 1), dtype=torch.int32),
        routing_weights=torch.ones((1, 1), dtype=torch.float32),
        distinct_expert_ids=(0,),
    )

    with pytest.raises(TransportError) as caught:
        group_worker_batches(
            batch,
            snapshot,
            RoundRobinSelector(),
            excluded=frozenset((worker_a.identity,)),
        )

    assert caught.value.code is TransportErrorCode.UNAVAILABLE
    assert caught.value.unavailable_expert_ids == (0,)
