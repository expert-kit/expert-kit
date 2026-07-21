"""Tests for bounded Routed-layer retry and final aggregation."""

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest
import torch

import expertkit_transport.routing.execute as execute_module
from expertkit_transport.batches import RoutedLayerBatch, WorkerBatch
from expertkit_transport.buffers import OutputPool
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.routing import (
    RoundRobinSelector,
    TopologyProvider,
    TopologySnapshot,
    WorkerConnection,
    WorkerIdentity,
    execute_routed_layer,
)
from expertkit_transport.transports.base import WorkerTransport


class ScriptedTransport(WorkerTransport):
    def __init__(self, outcomes: list[float | TransportError]) -> None:
        self.outcomes = outcomes
        self.calls: list[WorkerBatch] = []

    async def start(self) -> None:
        return None

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        self.calls.append(batch)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, TransportError):
            raise outcome
        output.fill_(outcome)

    async def close(self) -> None:
        return None


class BlockingTransport(WorkerTransport):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def start(self) -> None:
        return None

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        self.started.set()
        await self.gate.wait()

    async def close(self) -> None:
        return None


class FakeTopologyProvider(TopologyProvider):
    def __init__(
        self,
        initial: TopologySnapshot,
        refreshed: TopologySnapshot | None = None,
    ) -> None:
        self.initial = initial
        self.refreshed = refreshed or initial
        self.refresh_calls: list[tuple[int, int, float]] = []

    def current(self, instance_id: int) -> TopologySnapshot:
        assert instance_id == self.initial.instance_id
        return self.initial

    async def refresh(
        self,
        instance_id: int,
        *,
        observed_version: int,
        monotonic_deadline: float,
    ) -> TopologySnapshot:
        self.refresh_calls.append((instance_id, observed_version, monotonic_deadline))
        return self.refreshed


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def target(
    name: str,
    transport: WorkerTransport,
    *,
    max_batch_tokens: int = 8,
    max_active_batches: int = 2,
    max_pending_batches: int = 2,
) -> WorkerConnection:
    return WorkerConnection(
        identity=WorkerIdentity(name, f"{name}-start"),
        transport=transport,
        max_batch_tokens=max_batch_tokens,
        max_active_batches=max_active_batches,
        max_pending_batches=max_pending_batches,
    )


def routed_batch(
    expert_ids: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float16,
) -> RoutedLayerBatch:
    token_count = expert_ids.shape[0]
    distinct_expert_ids = tuple(sorted(set(expert_ids[expert_ids >= 0].tolist())))
    return RoutedLayerBatch(
        instance_id=7,
        layer_id=2,
        hidden_states=torch.zeros((token_count, 4), dtype=dtype),
        expert_ids=expert_ids,
        routing_weights=torch.where(
            expert_ids >= 0,
            torch.full(expert_ids.shape, 0.25, dtype=torch.float32),
            torch.zeros(expert_ids.shape, dtype=torch.float32),
        ),
        distinct_expert_ids=distinct_expert_ids,
    )


def pools_for(
    targets: tuple[WorkerConnection, ...],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[WorkerIdentity, OutputPool]:
    return {
        worker.identity: OutputPool(
            max_batch_tokens=8,
            hidden_dim=4,
            dtype=torch.float16,
            device="cpu",
            capacity=worker.max_in_flight,
            clock=clock,
        )
        for worker in targets
    }


async def close_pools(pools: dict[WorkerIdentity, OutputPool]) -> None:
    for pool in pools.values():
        await pool.close()


def run(coroutine: Awaitable[None]) -> None:
    asyncio.run(coroutine)


def test_single_complete_worker_bypasses_fp32_dispatch(monkeypatch) -> None:
    async def scenario() -> None:
        transport = ScriptedTransport([1.5])
        worker = target("worker-a", transport)
        snapshot = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        pools = pools_for((worker,))

        async def unexpected_dispatch(*args: object, **kwargs: object) -> tuple[object, ...]:
            raise AssertionError("a complete single-Worker result must bypass FP32 dispatch")

        monkeypatch.setattr(execute_module, "dispatch_once", unexpected_dispatch)
        result = await execute_module.execute_routed_layer(
            routed_batch(torch.tensor([[0], [0]], dtype=torch.int32)),
            FakeTopologyProvider(snapshot),
            RoundRobinSelector(),
            pools,
            monotonic_deadline=float("inf"),
        )

        assert result.dtype is torch.float16
        torch.testing.assert_close(result, torch.full((2, 4), 1.5, dtype=torch.float16))
        assert transport.calls[0].token_indices is None
        await close_pools(pools)

    run(scenario())


def test_retry_keeps_successes_once_and_prefers_replacement() -> None:
    async def scenario() -> None:
        busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        transport_a = ScriptedTransport([1.0])
        transport_b = ScriptedTransport([busy])
        transport_c = ScriptedTransport([2.0])
        worker_a = target("worker-a", transport_a)
        worker_b = target("worker-b", transport_b)
        worker_c = target("worker-c", transport_c)
        initial = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker_a,), (2, 1): (worker_b,)},
        )
        refreshed = TopologySnapshot(
            instance_id=7,
            version=12,
            routes={(2, 0): (worker_a,), (2, 1): (worker_c, worker_b)},
        )
        topology = FakeTopologyProvider(initial, refreshed)
        clock = FakeClock(10.0)
        pools = pools_for((worker_a, worker_b, worker_c), clock=clock)

        result = await execute_routed_layer(
            routed_batch(torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)),
            topology,
            RoundRobinSelector(),
            pools,
            monotonic_deadline=20.0,
            clock=clock,
            sleep=clock.sleep,
        )

        assert result.dtype == torch.float16
        torch.testing.assert_close(result, torch.full((2, 4), 3.0, dtype=torch.float16))
        assert len(transport_a.calls) == len(transport_b.calls) == len(transport_c.calls) == 1
        assert transport_c.calls[0].expert_ids.tolist() == [[-1, 1], [-1, 1]]
        assert topology.refresh_calls == [(7, 11, 20.0)]
        assert clock.sleeps == []
        await close_pools(pools)

    run(scenario())


def test_retry_rebuilds_only_failed_physical_chunk() -> None:
    async def scenario() -> None:
        busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        transport_a = ScriptedTransport([1.0, busy, 3.0])
        transport_c = ScriptedTransport([2.0])
        worker_a = target("worker-a", transport_a, max_batch_tokens=1)
        worker_c = target("worker-c", transport_c)
        initial = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker_a,)},
        )
        refreshed = TopologySnapshot(
            instance_id=7,
            version=12,
            routes={(2, 0): (worker_c,)},
        )
        pools = pools_for((worker_a, worker_c))

        result = await execute_routed_layer(
            routed_batch(torch.tensor([[0], [0], [0]], dtype=torch.int32)),
            FakeTopologyProvider(initial, refreshed),
            RoundRobinSelector(),
            pools,
            monotonic_deadline=float("inf"),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([[1.0] * 4, [2.0] * 4, [3.0] * 4], dtype=torch.float16),
        )
        assert len(transport_a.calls) == 3
        assert len(transport_c.calls) == 1
        assert transport_c.calls[0].token_indices.tolist() == [1]
        await close_pools(pools)

    run(scenario())


def test_nonretryable_failure_does_not_refresh_topology() -> None:
    async def scenario() -> None:
        invalid = TransportError(TransportErrorCode.INVALID_REQUEST, retryable=False)
        transport = ScriptedTransport([invalid])
        worker = target("worker-a", transport)
        snapshot = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        topology = FakeTopologyProvider(snapshot)
        pools = pools_for((worker,))

        with pytest.raises(TransportError) as caught:
            await execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                topology,
                RoundRobinSelector(),
                pools,
                monotonic_deadline=float("inf"),
            )

        assert caught.value is invalid
        assert topology.refresh_calls == []
        await close_pools(pools)

    run(scenario())


def test_same_process_retry_waits_once_when_no_replacement_exists() -> None:
    async def scenario() -> None:
        busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        transport = ScriptedTransport([busy, 4.0])
        worker = target("worker-a", transport)
        initial = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        refreshed = TopologySnapshot(
            instance_id=7,
            version=12,
            routes={(2, 0): (worker,)},
        )
        clock = FakeClock(10.0)
        pools = pools_for((worker,), clock=clock)

        result = await execute_routed_layer(
            routed_batch(torch.tensor([[0]], dtype=torch.int32)),
            FakeTopologyProvider(initial, refreshed),
            RoundRobinSelector(),
            pools,
            monotonic_deadline=20.0,
            same_worker_retry_delay_seconds=0.01,
            clock=clock,
            sleep=clock.sleep,
        )

        torch.testing.assert_close(result, torch.full((1, 4), 4.0, dtype=torch.float16))
        assert len(transport.calls) == 2
        assert clock.sleeps == [0.01]
        await close_pools(pools)

    run(scenario())


def test_same_process_retry_requires_useful_deadline_budget() -> None:
    async def scenario() -> None:
        busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        transport = ScriptedTransport([busy])
        worker = target("worker-a", transport)
        snapshot = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        topology = FakeTopologyProvider(snapshot)
        clock = FakeClock(10.0)
        pools = pools_for((worker,), clock=clock)

        with pytest.raises(TransportError) as caught:
            await execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                topology,
                RoundRobinSelector(),
                pools,
                monotonic_deadline=10.01,
                same_worker_retry_delay_seconds=0.02,
                clock=clock,
                sleep=clock.sleep,
            )

        assert caught.value is busy
        assert len(transport.calls) == 1
        assert len(topology.refresh_calls) == 1
        assert clock.sleeps == []
        await close_pools(pools)

    run(scenario())


def test_retry_stops_after_second_failed_attempt() -> None:
    async def scenario() -> None:
        first_busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        second_busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        transport = ScriptedTransport([first_busy, second_busy])
        worker = target("worker-a", transport)
        snapshot = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        topology = FakeTopologyProvider(snapshot)
        clock = FakeClock(10.0)
        pools = pools_for((worker,), clock=clock)

        with pytest.raises(TransportError) as caught:
            await execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                topology,
                RoundRobinSelector(),
                pools,
                monotonic_deadline=20.0,
                clock=clock,
                sleep=clock.sleep,
            )

        assert caught.value is second_busy
        assert len(transport.calls) == 2
        assert len(topology.refresh_calls) == 1
        assert clock.sleeps == [0.001]
        await close_pools(pools)

    run(scenario())


def test_retry_rejects_topology_older_than_draining_requirement() -> None:
    async def scenario() -> None:
        draining = TransportError(
            TransportErrorCode.DRAINING,
            retryable=True,
            min_topology_version=13,
        )
        transport = ScriptedTransport([draining])
        worker = target("worker-a", transport)
        initial = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        refreshed = TopologySnapshot(
            instance_id=7,
            version=12,
            routes={(2, 0): (worker,)},
        )
        pools = pools_for((worker,))

        with pytest.raises(TransportError) as caught:
            await execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                FakeTopologyProvider(initial, refreshed),
                RoundRobinSelector(),
                pools,
                monotonic_deadline=float("inf"),
            )

        assert caught.value.code is TransportErrorCode.UNAVAILABLE
        assert caught.value.min_topology_version == 13
        assert caught.value.unavailable_expert_ids == (0,)
        assert len(transport.calls) == 1
        await close_pools(pools)

    run(scenario())


def test_retry_returns_unavailable_without_waiting_for_future_route() -> None:
    async def scenario() -> None:
        busy = TransportError(TransportErrorCode.BUSY, retryable=True)
        transport = ScriptedTransport([busy])
        worker = target("worker-a", transport)
        initial = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        refreshed = TopologySnapshot(instance_id=7, version=12, routes={})
        topology = FakeTopologyProvider(initial, refreshed)
        pools = pools_for((worker,))

        with pytest.raises(TransportError) as caught:
            await execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                topology,
                RoundRobinSelector(),
                pools,
                monotonic_deadline=float("inf"),
            )

        assert caught.value.code is TransportErrorCode.UNAVAILABLE
        assert caught.value.retryable is True
        assert caught.value.unavailable_expert_ids == (0,)
        assert len(topology.refresh_calls) == 1
        assert len(transport.calls) == 1
        await close_pools(pools)

    run(scenario())


def test_cancellation_releases_output_pool() -> None:
    async def scenario() -> None:
        transport = BlockingTransport()
        worker = target("worker-a", transport)
        snapshot = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        pools = pools_for((worker,))
        execution = asyncio.create_task(
            execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                FakeTopologyProvider(snapshot),
                RoundRobinSelector(),
                pools,
                monotonic_deadline=float("inf"),
            )
        )
        await transport.started.wait()
        execution.cancel()

        with pytest.raises(asyncio.CancelledError):
            await execution
        async with pools[worker.identity].lease(monotonic_deadline=float("inf")):
            pass
        await close_pools(pools)

    run(scenario())


@pytest.mark.parametrize("delay", [0.0, -0.1, float("inf"), float("nan")])
def test_retry_delay_must_be_finite_and_positive(delay: float) -> None:
    async def scenario() -> None:
        transport = ScriptedTransport([1.0])
        worker = target("worker-a", transport)
        snapshot = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,)},
        )
        pools = pools_for((worker,))

        with pytest.raises(ValueError, match="finite and positive"):
            await execute_routed_layer(
                routed_batch(torch.tensor([[0]], dtype=torch.int32)),
                FakeTopologyProvider(snapshot),
                RoundRobinSelector(),
                pools,
                monotonic_deadline=float("inf"),
                same_worker_retry_delay_seconds=delay,
            )
        await close_pools(pools)

    run(scenario())
