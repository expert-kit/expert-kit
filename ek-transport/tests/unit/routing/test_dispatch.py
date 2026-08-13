"""Tests for bounded concurrent dispatch and FP32 aggregation."""

import asyncio
import threading

import pytest
import torch

import expertkit_transport.routing.dispatch as dispatch_module
from expertkit_transport import buffers as buffers_module
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


class _FailingRecordEvent:
    def record(self, stream: object) -> None:
        del stream
        raise RuntimeError("injected aggregation fence record failure")

    def synchronize(self) -> None:
        return None


class _ControlledEvent:
    def __init__(self, *, fail_synchronize: bool = False) -> None:
        self.fail_synchronize = fail_synchronize
        self.synchronize_started = threading.Event()
        self.allow_synchronize = threading.Event()
        self.record_count = 0
        self.synchronize_count = 0

    def record(self, stream: object) -> None:
        del stream
        self.record_count += 1

    def synchronize(self) -> None:
        self.synchronize_count += 1
        self.synchronize_started.set()
        if not self.allow_synchronize.wait(5):
            raise RuntimeError("test did not release aggregation fence")
        if self.fail_synchronize:
            raise RuntimeError("injected aggregation fence synchronize failure")


class _FakeStream:
    def wait_event(self, event: object) -> None:
        del event


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


def test_aggregation_fence_failure_retains_the_complete_routing_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = FakeTransport(1.0)
        worker = target("worker-a", transport)
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,), (2, 1): (worker,)},
        )
        plans = group_worker_batches(routed_batch(), topology, RoundRobinSelector())
        assert len(plans) == 1
        pools = pools_for((worker,))
        output_pool = pools[worker.identity]
        output_slot = output_pool._available[-1]  # type: ignore[attr-defined]
        output_slot.reuse_event = _FailingRecordEvent()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())
        accumulator = torch.zeros((3, 4), dtype=torch.float32)

        failures = await dispatch_once(
            plans,
            pools,
            accumulator,
            monotonic_deadline=float("inf"),
        )

        assert len(failures) == 1
        failure = failures[0]
        assert failure.plan is plans[0]
        assert not failure.error.retryable
        assert failure.error.unsafe_tensor_ownership
        assert failure.error.unsafe_output
        retained = buffers_module._QUARANTINED_OUTPUT_GRAPHS[-1]  # type: ignore[attr-defined]
        for owner in (
            accumulator,
            plans[0],
            plans[0].batch,
            transport,
            output_slot,
        ):
            assert any(candidate is owner for candidate in retained)
        retained_tensors = tuple(
            candidate for candidate in retained if isinstance(candidate, torch.Tensor)
        )
        assert any(tensor.dtype == torch.int64 for tensor in retained_tensors)
        assert sum(tensor.dtype == torch.float32 for tensor in retained_tensors) >= 2
        assert all(  # type: ignore[attr-defined]
            candidate is not output_slot for candidate in output_pool._available
        )
        assert any(  # type: ignore[attr-defined]
            candidate is output_slot for candidate in output_pool._quarantined
        )
        await output_pool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["to", "index_add"])
def test_aggregation_exception_waits_for_event_despite_repeated_cancellation(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = FakeTransport(1.0)
        worker = target("worker-a", transport)
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,), (2, 1): (worker,)},
        )
        plan = group_worker_batches(routed_batch(), topology, RoundRobinSelector())[0]
        output_pool = pools_for((worker,))[worker.identity]
        event = _ControlledEvent()
        output_slot = output_pool._available[-1]  # type: ignore[attr-defined]
        output_slot.reuse_event = event
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())

        if phase == "to":

            def fail_to(partial: torch.Tensor) -> torch.Tensor:
                del partial
                raise RuntimeError("injected to failure")

            monkeypatch.setattr(dispatch_module, "_to_fp32", fail_to)
        else:

            def fail_index_add(
                accumulator: torch.Tensor,
                token_indices: torch.Tensor,
                partial: torch.Tensor,
            ) -> None:
                del accumulator, token_indices, partial
                raise RuntimeError("injected index_add failure")

            monkeypatch.setattr(dispatch_module, "_index_add", fail_index_add)

        submission = asyncio.create_task(
            dispatch_module._dispatch_plan(  # type: ignore[attr-defined]
                plan,
                output_pool,
                torch.zeros((3, 4), dtype=torch.float32),
                float("inf"),
            )
        )
        while not event.synchronize_started.is_set():
            await asyncio.sleep(0)
        submission.cancel()
        await asyncio.sleep(0)
        submission.cancel()
        await asyncio.sleep(0)
        assert not submission.done()
        event.allow_synchronize.set()
        with pytest.raises(RuntimeError, match=f"injected {phase}"):
            await submission
        assert event.record_count == 1
        assert event.synchronize_count == 1
        assert len(output_pool._available) == output_pool.capacity  # type: ignore[attr-defined]
        assert any(  # type: ignore[attr-defined]
            candidate is output_slot for candidate in output_pool._available
        )
        output_slot.reuse_event = None
        await output_pool.close()

    asyncio.run(scenario())


def test_aggregation_exception_sync_failure_becomes_unsafe_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = FakeTransport(1.0)
        worker = target("worker-a", transport)
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,), (2, 1): (worker,)},
        )
        plan = group_worker_batches(routed_batch(), topology, RoundRobinSelector())[0]
        output_pool = pools_for((worker,))[worker.identity]
        event = _ControlledEvent(fail_synchronize=True)
        event.allow_synchronize.set()
        output_slot = output_pool._available[-1]  # type: ignore[attr-defined]
        output_slot.reuse_event = event
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())

        def fail_index_add(
            accumulator: torch.Tensor,
            token_indices: torch.Tensor,
            partial: torch.Tensor,
        ) -> None:
            del accumulator, token_indices, partial
            raise RuntimeError("injected index_add failure")

        monkeypatch.setattr(dispatch_module, "_index_add", fail_index_add)
        failure = await dispatch_module._dispatch_plan(  # type: ignore[attr-defined]
            plan,
            output_pool,
            torch.zeros((3, 4), dtype=torch.float32),
            float("inf"),
        )

        assert failure is not None
        assert not failure.error.retryable
        assert failure.error.unsafe_tensor_ownership
        assert failure.error.unsafe_output
        assert "could not be proven" in failure.error.diagnostic
        assert all(  # type: ignore[attr-defined]
            candidate is not output_slot for candidate in output_pool._available
        )
        assert any(  # type: ignore[attr-defined]
            candidate is output_slot for candidate in output_pool._quarantined
        )
        await output_pool.close()

    asyncio.run(scenario())


def test_successful_aggregation_does_not_host_synchronize_the_reuse_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = FakeTransport(1.0)
        worker = target("worker-a", transport)
        topology = TopologySnapshot(
            instance_id=7,
            version=11,
            routes={(2, 0): (worker,), (2, 1): (worker,)},
        )
        plan = group_worker_batches(routed_batch(), topology, RoundRobinSelector())[0]
        output_pool = pools_for((worker,))[worker.identity]
        event = _ControlledEvent()
        output_slot = output_pool._available[-1]  # type: ignore[attr-defined]
        output_slot.reuse_event = event
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FakeStream())

        failure = await dispatch_module._dispatch_plan(  # type: ignore[attr-defined]
            plan,
            output_pool,
            torch.zeros((3, 4), dtype=torch.float32),
            float("inf"),
        )

        assert failure is None
        assert event.record_count == 1
        assert event.synchronize_count == 0
        output_slot.reuse_event = None
        await output_pool.close()

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
