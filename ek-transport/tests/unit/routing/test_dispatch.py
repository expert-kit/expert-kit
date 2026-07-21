"""Tests for bounded concurrent dispatch and FP32 aggregation."""

import asyncio

import pytest
import torch

from expertkit_transport.batches import RoutedLayerBatch, WorkerBatch
from expertkit_transport.buffers import OutputPool
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.routing import (
    RoundRobinSelector,
    TopologySnapshot,
    WorkerConnection,
    WorkerIdentity,
    dispatch_once,
    group_worker_batches,
)
from expertkit_transport.transports.base import WorkerTransport


class FakeTransport(WorkerTransport):
    def __init__(
        self,
        value: float,
        *,
        started: asyncio.Event | None = None,
        gate: asyncio.Event | None = None,
        error: TransportError | None = None,
    ) -> None:
        self.value = value
        self.started = started
        self.gate = gate
        self.error = error

    async def start(self) -> None:
        return None

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        output.fill_(self.value)

    async def close(self) -> None:
        return None


def target(name: str, transport: WorkerTransport) -> WorkerConnection:
    return WorkerConnection(
        identity=WorkerIdentity(name, f"{name}-start"),
        transport=transport,
        max_batch_tokens=8,
        max_active_batches=1,
        max_pending_batches=1,
    )


def routed_batch() -> RoutedLayerBatch:
    return RoutedLayerBatch(
        instance_id=7,
        layer_id=2,
        hidden_states=torch.zeros((3, 4), dtype=torch.float16),
        expert_ids=torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.int32),
        routing_weights=torch.ones((3, 2), dtype=torch.float32),
        distinct_expert_ids=(0, 1),
    )


def pools_for(targets: tuple[WorkerConnection, ...]) -> dict[WorkerIdentity, OutputPool]:
    return {
        worker.identity: OutputPool(
            max_batch_tokens=8,
            hidden_dim=4,
            dtype=torch.float16,
            device="cpu",
            capacity=worker.max_in_flight,
        )
        for worker in targets
    }


def test_dispatches_workers_concurrently_and_aggregates_in_fp32() -> None:
    async def scenario() -> None:
        started_a = asyncio.Event()
        started_b = asyncio.Event()
        gate = asyncio.Event()
        worker_a = target("worker-a", FakeTransport(1.0, started=started_a, gate=gate))
        worker_b = target("worker-b", FakeTransport(2.0, started=started_b, gate=gate))
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker_a,), (2, 1): (worker_b,)},
        )
        plans = group_worker_batches(routed_batch(), topology, RoundRobinSelector())
        pools = pools_for((worker_a, worker_b))
        accumulator = torch.zeros((3, 4), dtype=torch.float32)

        dispatch = asyncio.create_task(
            dispatch_once(plans, pools, accumulator, monotonic_deadline=float("inf"))
        )
        await started_a.wait()
        await started_b.wait()
        gate.set()

        assert await dispatch == ()
        torch.testing.assert_close(
            accumulator,
            torch.tensor(
                [[3.0] * 4, [3.0] * 4, [1.0] * 4],
                dtype=torch.float32,
            ),
        )
        for pool in pools.values():
            await pool.close()

    asyncio.run(scenario())


def test_failed_contribution_is_not_aggregated_or_lost() -> None:
    async def scenario() -> None:
        busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        worker_a = target("worker-a", FakeTransport(1.0))
        worker_b = target("worker-b", FakeTransport(2.0, error=busy))
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker_a,), (2, 1): (worker_b,)},
        )
        plans = group_worker_batches(routed_batch(), topology, RoundRobinSelector())
        pools = pools_for((worker_a, worker_b))
        accumulator = torch.zeros((3, 4), dtype=torch.float32)

        failures = await dispatch_once(
            plans,
            pools,
            accumulator,
            monotonic_deadline=float("inf"),
        )

        assert len(failures) == 1
        assert failures[0].plan.target.identity == worker_b.identity
        assert failures[0].error is busy
        torch.testing.assert_close(
            accumulator,
            torch.tensor(
                [[1.0] * 4, [1.0] * 4, [1.0] * 4],
                dtype=torch.float32,
            ),
        )
        for pool in pools.values():
            await pool.close()

    asyncio.run(scenario())


def test_cancellation_returns_output_buffer_after_submit_cleanup() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        gate = asyncio.Event()
        worker = target("worker-a", FakeTransport(1.0, started=started, gate=gate))
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,), (2, 1): (worker,)},
        )
        plans = group_worker_batches(routed_batch(), topology, RoundRobinSelector())
        pools = pools_for((worker,))
        accumulator = torch.zeros((3, 4), dtype=torch.float32)

        dispatch = asyncio.create_task(
            dispatch_once(
                plans,
                pools,
                accumulator,
                monotonic_deadline=float("inf"),
            )
        )
        await started.wait()
        dispatch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dispatch

        pool = pools[worker.identity]
        async with pool.lease(monotonic_deadline=float("inf")):
            pass
        await pool.close()

    asyncio.run(scenario())
