"""Tests for immutable ready-route Topology snapshots."""

import pytest

from expertkit_transport.contracts import (
    OutputBufferProvider,
    PreparedOutput,
    WorkerBatch,
    WorkerTransport,
)
from expertkit_transport.orchestration import TopologySnapshot, WorkerIdentity, WorkerTarget


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

    assert snapshot.layer_routes(2)[3] == (worker,)
    with pytest.raises(TypeError):
        snapshot.routes[(2, 4)] = (worker,)  # type: ignore[index]


def test_snapshot_rejects_inconsistent_metadata_for_one_process() -> None:
    worker = target("worker-a")
    inconsistent = WorkerTarget(
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


def test_worker_target_requires_positive_published_limits() -> None:
    with pytest.raises(ValueError, match="max_batch_tokens"):
        target("worker-a", max_batch_tokens=0)
