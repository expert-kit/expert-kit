"""Tests for the reconnectable Worker weight-control stream."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine, Iterable
from typing import Any

import pytest
from expertkit_transport._proto.ek.control.v2 import weight_control_pb2
from expertkit_transport.contracts import (
    ReceivedWorkerBatch,
    WorkerBatchReceiver,
    WorkerPositionBuffers,
    WorkerPositionSpec,
)

from expertkit_worker.control import (
    ExpertStateReporter,
    WeightControlDrainError,
    WeightControlSession,
)
from expertkit_worker.weights import (
    ExpertState,
    ExpertStateChange,
    ExpertStateKind,
    TargetExpert,
)
from expertkit_worker.weights.dram_cache import WeightKey


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _reporter() -> ExpertStateReporter:
    return ExpertStateReporter(
        num_layers=2,
        experts_per_layer=8,
        max_updates=64,
        max_delay_ms=1,
    )


def _target_part(
    generation: int,
    experts: tuple[tuple[int, int], ...],
) -> weight_control_pb2.ControllerWeightMessage:
    return weight_control_pb2.ControllerWeightMessage(
        targets=weight_control_pb2.TargetExpertListPart(
            placement_generation=generation,
            part_index=0,
            part_count=1,
            experts=[
                weight_control_pb2.TargetExpert(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    target_device="cuda:0",
                )
                for layer_id, expert_id in experts
            ],
        )
    )


def _drain_part(
    *,
    drain_id: int,
    generation: int,
    experts: tuple[tuple[int, int], ...],
    stop_all: bool = False,
) -> weight_control_pb2.ControllerWeightMessage:
    return weight_control_pb2.ControllerWeightMessage(
        drain=weight_control_pb2.DrainAuthorizationPart(
            drain_id=drain_id,
            placement_generation=generation,
            min_topology_version=9,
            stop_accepting_all_computation=stop_all,
            part_index=0,
            part_count=1,
            experts=[
                {"layer_id": layer_id, "expert_id": expert_id} for layer_id, expert_id in experts
            ],
        )
    )


class FakeManager:
    def __init__(
        self,
        reporter: ExpertStateReporter,
        *,
        states: tuple[ExpertState, ...] = (),
    ) -> None:
        self.reporter = reporter
        self.generation = 0
        self.states = states
        self.targets: tuple[TargetExpert, ...] = ()
        self.removals: list[tuple[int, tuple[WeightKey, ...]]] = []
        self.shutting_down = False

    async def begin_shutdown(self) -> bool:
        if self.shutting_down:
            return False
        self.shutting_down = True
        return True

    async def apply_targets(
        self,
        placement_generation: int,
        targets: Iterable[TargetExpert],
    ) -> bool:
        resolved = tuple(targets)
        if placement_generation == self.generation:
            assert resolved == self.targets
            return False
        self.generation = placement_generation
        self.targets = resolved
        return True

    async def snapshot(self) -> tuple[int, tuple[ExpertState, ...]]:
        return self.generation, self.states

    async def remove_after_drain(
        self,
        placement_generation: int,
        keys: Iterable[WeightKey],
    ) -> bool:
        resolved = tuple(keys)
        self.removals.append((placement_generation, resolved))
        retained = {state.key: state for state in self.states}
        for key in resolved:
            removed = ExpertState(key, ExpertStateKind.REMOVED)
            retained[key] = removed
            self.reporter.record(ExpertStateChange(placement_generation, removed))
        self.states = tuple(retained[key] for key in sorted(retained))
        return placement_generation == self.generation


class FakeReceiver(WorkerBatchReceiver):
    def __init__(self) -> None:
        self.cleared: list[tuple[tuple[int, int], ...]] = []
        self.begun: list[tuple[tuple[tuple[int, int], ...], int, bool]] = []
        self.expert_idle: list[tuple[tuple[int, int], ...]] = []
        self.all_idle = 0
        self.wait_error: Exception | None = None

    async def take(self) -> ReceivedWorkerBatch:
        raise NotImplementedError

    def allocate_position_buffers(self, spec: WorkerPositionSpec) -> WorkerPositionBuffers:
        raise NotImplementedError

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        self.begun.append((tuple(experts), min_topology_version, stop_all))

    async def clear_expert_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        self.cleared.append(tuple(experts))

    async def wait_experts_idle(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        monotonic_deadline: float,
    ) -> None:
        assert monotonic_deadline == float("inf")
        if self.wait_error is not None:
            raise self.wait_error
        self.expert_idle.append(tuple(experts))

    async def wait_all_idle(self, *, monotonic_deadline: float) -> None:
        assert monotonic_deadline > 0
        if self.wait_error is not None:
            raise self.wait_error
        self.all_idle += 1

    async def close(self) -> None:
        return None


class PlacementAndUpdateRpc:
    def __init__(self, reporter: ExpertStateReporter) -> None:
        self.reporter = reporter
        self.sent: list[weight_control_pb2.WorkerWeightMessage] = []

    async def sync_weights(
        self,
        requests: AsyncIterator[weight_control_pb2.WorkerWeightMessage],
    ) -> AsyncIterator[weight_control_pb2.ControllerWeightMessage]:
        self.sent.append(await anext(requests))
        yield _target_part(3, ((0, 1),))
        full_state = await anext(requests)
        self.sent.append(full_state)
        yield weight_control_pb2.ControllerWeightMessage(
            state_ack=weight_control_pb2.StateReportAck(
                report_sequence=full_state.full_state.report_sequence
            )
        )

        self.reporter.record(
            ExpertStateChange(
                3,
                ExpertState(WeightKey(0, 1), ExpertStateKind.READY),
            )
        )
        update = await anext(requests)
        self.sent.append(update)
        yield weight_control_pb2.ControllerWeightMessage(
            state_ack=weight_control_pb2.StateReportAck(
                report_sequence=update.state_updates.report_sequence
            )
        )


def test_session_opens_applies_targets_snapshots_and_sends_updates() -> None:
    async def scenario() -> None:
        reporter = _reporter()
        ready = ExpertState(WeightKey(0, 1), ExpertStateKind.READY)
        manager = FakeManager(reporter, states=(ready,))
        receiver = FakeReceiver()
        session = WeightControlSession(
            worker_id="worker-0",
            start_id="start-0",
            max_experts=4,
            shutdown_grace_secs=30,
            manager=manager,
            reporter=reporter,
            receiver=receiver,
        )
        rpc = PlacementAndUpdateRpc(reporter)

        await session.run_once(rpc)

        assert [message.WhichOneof("message") for message in rpc.sent] == [
            "open",
            "full_state",
            "state_updates",
        ]
        assert manager.targets[0].key == WeightKey(0, 1)
        assert rpc.sent[0].open.worker_id == "worker-0"
        assert rpc.sent[1].full_state.placement_generation == 3
        assert rpc.sent[2].state_updates.experts[0].state == weight_control_pb2.EXPERT_READY
        assert (0, 1) in {key for call in receiver.cleared for key in call}
        assert reporter.acknowledge(2) is False
        await session.close()

    run(scenario())


class DrainRpc:
    def __init__(
        self,
        *,
        stop_all: bool = False,
        wait_forever_after_drain: bool = False,
    ) -> None:
        self.stop_all = stop_all
        self.wait_forever_after_drain = wait_forever_after_drain
        self.sent: list[weight_control_pb2.WorkerWeightMessage] = []

    async def sync_weights(
        self,
        requests: AsyncIterator[weight_control_pb2.WorkerWeightMessage],
    ) -> AsyncIterator[weight_control_pb2.ControllerWeightMessage]:
        self.sent.append(await anext(requests))
        yield _target_part(4, ())
        self.sent.append(await anext(requests))
        experts = () if self.stop_all else ((0, 2),)
        yield _drain_part(
            drain_id=12,
            generation=4,
            experts=experts,
            stop_all=self.stop_all,
        )
        if self.wait_forever_after_drain:
            await asyncio.Event().wait()
        else:
            if not self.stop_all:
                self.sent.append(await anext(requests))
            self.sent.append(await anext(requests))


def test_drain_reports_removed_before_completion() -> None:
    async def scenario() -> None:
        reporter = _reporter()
        manager = FakeManager(
            reporter,
            states=(ExpertState(WeightKey(0, 2), ExpertStateKind.READY),),
        )
        receiver = FakeReceiver()
        session = WeightControlSession(
            worker_id="worker-0",
            start_id="start-0",
            max_experts=4,
            shutdown_grace_secs=30,
            manager=manager,
            reporter=reporter,
            receiver=receiver,
        )
        rpc = DrainRpc()

        await session.run_once(rpc)

        assert [message.WhichOneof("message") for message in rpc.sent] == [
            "open",
            "full_state",
            "state_updates",
            "drain_complete",
        ]
        assert rpc.sent[2].state_updates.experts[0].state == weight_control_pb2.EXPERT_REMOVED
        assert rpc.sent[3].drain_complete.drain_id == 12
        assert receiver.begun == [(((0, 2),), 9, False)]
        assert receiver.expert_idle == [((0, 2),)]
        assert manager.removals == [(4, (WeightKey(0, 2),))]
        await session.close()

    run(scenario())


class ReplayRpc:
    def __init__(self) -> None:
        self.sent: list[weight_control_pb2.WorkerWeightMessage] = []

    async def sync_weights(
        self,
        requests: AsyncIterator[weight_control_pb2.WorkerWeightMessage],
    ) -> AsyncIterator[weight_control_pb2.ControllerWeightMessage]:
        self.sent.append(await anext(requests))
        yield _target_part(4, ())
        self.sent.append(await anext(requests))
        self.sent.append(await anext(requests))


def test_reconnect_sends_a_new_snapshot_and_replays_drain_completion() -> None:
    async def scenario() -> None:
        reporter = _reporter()
        manager = FakeManager(
            reporter,
            states=(ExpertState(WeightKey(0, 2), ExpertStateKind.READY),),
        )
        receiver = FakeReceiver()
        session = WeightControlSession(
            worker_id="worker-0",
            start_id="start-0",
            max_experts=4,
            shutdown_grace_secs=30,
            manager=manager,
            reporter=reporter,
            receiver=receiver,
        )
        await session.run_once(DrainRpc())
        replay = ReplayRpc()

        await session.run_once(replay)

        assert [message.WhichOneof("message") for message in replay.sent] == [
            "open",
            "full_state",
            "drain_complete",
        ]
        assert replay.sent[-1].drain_complete.drain_id == 12
        await session.close()

    run(scenario())


def test_whole_worker_drain_waits_for_every_admitted_batch() -> None:
    async def scenario() -> None:
        reporter = _reporter()
        manager = FakeManager(reporter)
        receiver = FakeReceiver()
        session = WeightControlSession(
            worker_id="worker-0",
            start_id="start-0",
            max_experts=4,
            shutdown_grace_secs=30,
            manager=manager,
            reporter=reporter,
            receiver=receiver,
        )
        rpc = DrainRpc(stop_all=True)

        assert await session.begin_shutdown() is True
        assert await session.begin_shutdown() is False
        await session.run_once(rpc)
        await session.wait_shutdown_drained()

        assert manager.shutting_down is True
        assert receiver.all_idle == 1
        assert receiver.expert_idle == []
        assert rpc.sent[-1].drain_complete.drain_id == 12
        await session.close()

    run(scenario())


def test_drain_failure_terminates_the_stream_attempt() -> None:
    async def scenario() -> None:
        reporter = _reporter()
        manager = FakeManager(reporter)
        manager.generation = 4
        receiver = FakeReceiver()
        receiver.wait_error = TimeoutError("drain timed out")
        session = WeightControlSession(
            worker_id="worker-0",
            start_id="start-0",
            max_experts=4,
            shutdown_grace_secs=30,
            manager=manager,
            reporter=reporter,
            receiver=receiver,
        )

        with pytest.raises(WeightControlDrainError) as caught:
            await session.run_once(DrainRpc(wait_forever_after_drain=True))

        assert isinstance(caught.value.__cause__, TimeoutError)
        await session.close()

    run(scenario())
