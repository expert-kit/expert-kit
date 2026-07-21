"""Tests for immutable ready-route Topology snapshots."""

import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.routing import TopologySnapshot, WorkerConnection, WorkerIdentity
from expertkit_transport.transports.base import WorkerTransport


class FakeTransport(WorkerTransport):
    async def start(self) -> None:
        return None

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def target(name: str, *, max_batch_tokens: int = 8) -> WorkerConnection:
    return WorkerConnection(
        identity=WorkerIdentity(name, f"{name}-start"),
        transport=FakeTransport(),
        max_batch_tokens=max_batch_tokens,
        max_active_batches=1,
        max_pending_batches=1,
    )


def test_snapshot_copies_route_mapping_and_replica_sequences() -> None:
    worker = target("worker-a")
    replicas = [worker]
    routes = {(2, 3): replicas}

    snapshot = TopologySnapshot(instance_id=7, version=11, routes=routes)
    replicas.clear()
    routes.clear()

    first = snapshot.layer_routes(2)
    second = snapshot.layer_routes(2)
    assert first[3] == (worker,)
    assert first is second
    with pytest.raises(TypeError):
        snapshot.routes[(2, 4)] = (worker,)  # type: ignore[index]


def test_snapshot_rejects_inconsistent_metadata_for_one_process() -> None:
    worker = target("worker-a")
    inconsistent = WorkerConnection(
        identity=worker.identity,
        transport=worker.transport,
        max_batch_tokens=4,
        max_active_batches=1,
        max_pending_batches=1,
    )

    with pytest.raises(ValueError, match="consistent route metadata"):
        TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,), (2, 1): (inconsistent,)},
        )


def test_worker_connection_requires_positive_published_limits() -> None:
    with pytest.raises(ValueError, match="max_batch_tokens"):
        target("worker-a", max_batch_tokens=0)
