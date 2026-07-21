"""Controller weight-placement, state-reporting, and drain stream."""

from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from expertkit_proto.ek.control.v2 import weight_control_pb2
from expertkit_transport.contracts import WorkerBatchReceiver

from expertkit_worker.control.parts import (
    DrainAuthorization,
    DrainAuthorizationAssembler,
    TargetListAssembler,
)
from expertkit_worker.control.state_reporter import ExpertStateReporter
from expertkit_worker.weights import ExpertState, ExpertStateKind, TargetExpert
from expertkit_worker.weights.dram_cache import WeightKey

_OUTBOUND_GROUPS = 4
_COMPLETED_DRAIN_HISTORY = 64


class WeightControlDrainError(RuntimeError):
    """Indicate that an authorized local drain could not finish safely."""


class _WeightManager(Protocol):
    async def begin_shutdown(self) -> bool:
        """Stop loading and reject newly assigned experts."""

    async def apply_targets(
        self,
        placement_generation: int,
        targets: Iterable[TargetExpert],
    ) -> bool:
        """Apply one complete placement generation."""

    async def snapshot(self) -> tuple[int, tuple[ExpertState, ...]]:
        """Return a consistent current expert-state snapshot."""

    async def remove_after_drain(
        self,
        placement_generation: int,
        keys: Iterable[WeightKey],
        *,
        whole_worker_shutdown: bool = False,
    ) -> bool:
        """Make drained experts unavailable after Transport admission is idle."""


class _WeightControlRpc(Protocol):
    def sync_weights(
        self,
        requests: AsyncIterator[weight_control_pb2.WorkerWeightMessage],
    ) -> AsyncIterator[weight_control_pb2.ControllerWeightMessage]:
        """Open one bidirectional weight-control stream."""


@dataclass(slots=True)
class _StreamRun:
    outbound: asyncio.Queue[tuple[weight_control_pb2.WorkerWeightMessage, ...]]
    snapshot_ready: asyncio.Event


@dataclass(frozen=True, slots=True)
class _ActiveDrain:
    authorization: DrainAuthorization
    task: asyncio.Task[None]


class WeightControlSession:
    """Preserve placement, report, and drain state across stream reconnects."""

    def __init__(
        self,
        *,
        worker_id: str,
        start_id: str,
        max_experts: int,
        shutdown_grace_secs: float,
        manager: _WeightManager,
        reporter: ExpertStateReporter,
        receiver: WorkerBatchReceiver,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (("worker_id", worker_id), ("start_id", start_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if isinstance(max_experts, bool) or not isinstance(max_experts, int) or max_experts <= 0:
            raise ValueError("max_experts must be a positive integer")
        if (
            isinstance(shutdown_grace_secs, bool)
            or not isinstance(shutdown_grace_secs, int | float)
            or shutdown_grace_secs <= 0
        ):
            raise ValueError("shutdown_grace_secs must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._worker_id = worker_id
        self._start_id = start_id
        self._shutdown_grace_secs = float(shutdown_grace_secs)
        self._manager = manager
        self._reporter = reporter
        self._receiver = receiver
        self._clock = clock
        self._targets = TargetListAssembler(max_experts)
        self._drains = DrainAuthorizationAssembler(max_experts)
        self._current_placement_generation = 0
        self._current_target_keys: set[WeightKey] = set()
        self._drain_transition = asyncio.Lock()
        self._run_lock = asyncio.Lock()
        self._report_condition = asyncio.Condition()
        self._last_enqueued_report_sequence = 0
        self._active_drains: dict[int, _ActiveDrain] = {}
        self._completed_drains: OrderedDict[int, DrainAuthorization] = OrderedDict()
        self._superseded_drains: OrderedDict[int, DrainAuthorization] = OrderedDict()
        self._completion_changed = asyncio.Event()
        self._drain_failed = asyncio.Event()
        self._drain_error: Exception | None = None
        self._shutdown_drained = asyncio.Event()
        self._shutdown_completion_sent = asyncio.Event()
        self._shutdown_deadline: float | None = None
        self._closed = False

    async def run_once(self, rpc: _WeightControlRpc) -> None:
        """Run one stream attempt while retaining process-lifetime state."""

        if self._closed:
            raise RuntimeError("Weight control session is closed")
        async with self._run_lock:
            if self._closed:
                raise RuntimeError("Weight control session is closed")
            run = _StreamRun(
                outbound=asyncio.Queue(maxsize=_OUTBOUND_GROUPS),
                snapshot_ready=asyncio.Event(),
            )
            tasks = {
                "responses": asyncio.create_task(
                    self._consume_responses(rpc, run),
                    name="weight-control-responses",
                ),
                "updates": asyncio.create_task(
                    self._pump_updates(run),
                    name="weight-control-updates",
                ),
                "completions": asyncio.create_task(
                    self._pump_completions(run),
                    name="weight-control-drain-completions",
                ),
                "drain_failure": asyncio.create_task(
                    self._wait_drain_failure(),
                    name="weight-control-drain-failure",
                ),
            }
            try:
                done, _pending = await asyncio.wait(
                    tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if tasks["drain_failure"] in done:
                    error = tasks["drain_failure"].result()
                    raise WeightControlDrainError("authorized drain failed") from error
                for name in ("updates", "completions"):
                    if tasks[name] in done:
                        await tasks[name]
                        raise RuntimeError(f"{name} sender stopped unexpectedly")
                if tasks["responses"] in done:
                    await tasks["responses"]
                    return
            finally:
                for task in tasks.values():
                    task.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def wait_shutdown_drained(self) -> None:
        """Wait for a whole-Worker drain to finish its local safety steps."""

        await self._shutdown_drained.wait()

    @property
    def shutdown_deadline(self) -> float | None:
        """Return the fixed overall shutdown deadline after shutdown begins."""

        return self._shutdown_deadline

    async def wait_shutdown_completion_sent(self) -> None:
        """Wait until whole-Worker drain completion enters the outgoing stream."""

        await self._shutdown_completion_sent.wait()

    async def begin_shutdown(self) -> bool:
        """Stop weight loading while keeping current ready experts available."""

        if self._shutdown_deadline is None:
            self._shutdown_deadline = self._clock() + self._shutdown_grace_secs
        return await self._manager.begin_shutdown()

    async def close(self) -> None:
        """Cancel outstanding drains and reject future stream attempts."""

        if self._closed:
            return
        self._closed = True
        tasks = tuple(active.task for active in self._active_drains.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume_responses(self, rpc: _WeightControlRpc, run: _StreamRun) -> None:
        async for response in rpc.sync_weights(self._requests(run)):
            message_kind = response.WhichOneof("message")
            if message_kind == "targets":
                placement = self._targets.add(response.targets)
                if placement is not None:
                    await self._apply_targets(
                        placement.placement_generation, placement.experts, run
                    )
            elif message_kind == "state_ack":
                self._reporter.acknowledge(response.state_ack.report_sequence)
            elif message_kind == "drain":
                authorization = self._drains.add(response.drain)
                if authorization is not None:
                    self._start_drain(authorization)
            else:
                raise ValueError("Controller weight message has no supported payload")

    async def _requests(
        self,
        run: _StreamRun,
    ) -> AsyncIterator[weight_control_pb2.WorkerWeightMessage]:
        yield weight_control_pb2.WorkerWeightMessage(
            open=weight_control_pb2.OpenWeightStream(
                worker_id=self._worker_id,
                start_id=self._start_id,
            )
        )
        while True:
            group = await run.outbound.get()
            await self._clear_ready_expert_drains(group)
            for message in group:
                if message.WhichOneof("message") == "drain_complete":
                    authorization = self._completed_drains.get(message.drain_complete.drain_id)
                    if authorization is not None and authorization.stop_accepting_all_computation:
                        self._shutdown_completion_sent.set()
                yield message

    async def _apply_targets(
        self,
        placement_generation: int,
        targets: tuple[TargetExpert, ...],
        run: _StreamRun,
    ) -> None:
        target_keys = {target.key for target in targets}
        superseded_tasks: list[asyncio.Task[None]] = []
        async with self._drain_transition:
            if placement_generation > self._current_placement_generation:
                for active in tuple(self._active_drains.values()):
                    authorization = active.authorization
                    if (
                        not authorization.stop_accepting_all_computation
                        and authorization.placement_generation < placement_generation
                    ):
                        self._remember_superseded(authorization)
                        active.task.cancel()
                        superseded_tasks.append(active.task)

            applied = await self._manager.apply_targets(placement_generation, targets)
            self._current_placement_generation = placement_generation
            self._current_target_keys = target_keys
            skip_snapshot = not applied and run.snapshot_ready.is_set()
            if not skip_snapshot:
                boundary = self._reporter.mark_changes()
                observed_generation, states = await self._manager.snapshot()
                if observed_generation != placement_generation:
                    raise RuntimeError("Weight Manager snapshot generation does not match targets")
                ready_targets = tuple(
                    (state.key.layer_id, state.key.expert_id)
                    for state in states
                    if state.state is ExpertStateKind.READY
                    and state.key in self._current_target_keys
                )
                if ready_targets:
                    await self._receiver.clear_expert_drains(ready_targets)

        if superseded_tasks:
            await asyncio.gather(*superseded_tasks, return_exceptions=True)
        if skip_snapshot:
            return

        if not run.snapshot_ready.is_set():
            group = self._reporter.full_state_parts(
                observed_generation,
                states,
                discard_pending_through=boundary,
            )
            await self._enqueue_reports(run, group)
            run.snapshot_ready.set()

    async def _pump_updates(self, run: _StreamRun) -> None:
        await run.snapshot_ready.wait()
        while True:
            message = await self._reporter.take_updates()
            if message is not None:
                await self._enqueue_reports(run, (message,))

    async def _pump_completions(self, run: _StreamRun) -> None:
        await run.snapshot_ready.wait()
        sent: set[int] = set()
        while True:
            pending_ids = tuple(
                drain_id for drain_id in self._completed_drains if drain_id not in sent
            )
            if pending_ids:
                report_sequence = await self._reporter.flush()
                await self._wait_report_enqueued(report_sequence)
                group = tuple(
                    weight_control_pb2.WorkerWeightMessage(
                        drain_complete=weight_control_pb2.DrainComplete(drain_id=drain_id)
                    )
                    for drain_id in pending_ids
                )
                await run.outbound.put(group)
                sent.update(pending_ids)
                continue

            self._completion_changed.clear()
            if any(drain_id not in sent for drain_id in self._completed_drains):
                continue
            await self._completion_changed.wait()

    async def _enqueue_reports(
        self,
        run: _StreamRun,
        group: tuple[weight_control_pb2.WorkerWeightMessage, ...],
    ) -> None:
        await run.outbound.put(group)
        sequences = tuple(self._report_sequence(message) for message in group)
        report_sequence = max(
            (sequence for sequence in sequences if sequence is not None), default=0
        )
        if report_sequence:
            async with self._report_condition:
                self._last_enqueued_report_sequence = max(
                    self._last_enqueued_report_sequence,
                    report_sequence,
                )
                self._report_condition.notify_all()

    @staticmethod
    def _report_sequence(message: weight_control_pb2.WorkerWeightMessage) -> int | None:
        message_kind = message.WhichOneof("message")
        if message_kind == "full_state":
            return message.full_state.report_sequence
        if message_kind == "state_updates":
            return message.state_updates.report_sequence
        return None

    async def _wait_report_enqueued(self, report_sequence: int) -> None:
        async with self._report_condition:
            while self._last_enqueued_report_sequence < report_sequence:
                await self._report_condition.wait()

    async def _clear_ready_expert_drains(
        self,
        group: tuple[weight_control_pb2.WorkerWeightMessage, ...],
    ) -> None:
        ready: set[tuple[int, int]] = set()
        for message in group:
            message_kind = message.WhichOneof("message")
            if message_kind == "full_state":
                states = message.full_state.experts
            elif message_kind == "state_updates":
                states = message.state_updates.experts
            else:
                continue
            ready.update(
                (state.layer_id, state.expert_id)
                for state in states
                if state.state == weight_control_pb2.EXPERT_READY
                and WeightKey(state.layer_id, state.expert_id) in self._current_target_keys
            )
        if ready:
            await self._receiver.clear_expert_drains(sorted(ready))

    def _start_drain(self, authorization: DrainAuthorization) -> None:
        superseded = self._superseded_drains.get(authorization.drain_id)
        if superseded is not None:
            if superseded != authorization:
                raise ValueError("superseded drain ID was repeated with different content")
            return
        completed = self._completed_drains.get(authorization.drain_id)
        if completed is not None:
            if completed != authorization:
                raise ValueError("completed drain ID was repeated with different content")
            self._completion_changed.set()
            return
        active = self._active_drains.get(authorization.drain_id)
        if active is not None:
            if active.authorization != authorization:
                raise ValueError("active drain ID was repeated with different content")
            return
        if (
            not authorization.stop_accepting_all_computation
            and authorization.placement_generation < self._current_placement_generation
        ):
            self._remember_superseded(authorization)
            return
        task = asyncio.create_task(
            self._run_drain(authorization),
            name=f"weight-control-drain-{authorization.drain_id}",
        )
        self._active_drains[authorization.drain_id] = _ActiveDrain(authorization, task)

    async def _run_drain(self, authorization: DrainAuthorization) -> None:
        keys = authorization.experts
        wire_keys = tuple((key.layer_id, key.expert_id) for key in keys)
        try:
            async with self._drain_transition:
                if self._is_superseded(authorization):
                    self._remember_superseded(authorization)
                    return
                await self._receiver.begin_drain(
                    wire_keys,
                    min_topology_version=authorization.min_topology_version,
                    stop_all=authorization.stop_accepting_all_computation,
                )
            if authorization.stop_accepting_all_computation:
                deadline = self._shutdown_deadline
                if deadline is None:
                    deadline = self._clock() + self._shutdown_grace_secs
                    self._shutdown_deadline = deadline
                await self._receiver.wait_all_idle(monotonic_deadline=deadline)
            else:
                await self._receiver.wait_experts_idle(
                    wire_keys,
                    monotonic_deadline=math.inf,
                )
            async with self._drain_transition:
                if self._is_superseded(authorization):
                    self._remember_superseded(authorization)
                    return
                removed = await self._manager.remove_after_drain(
                    authorization.placement_generation,
                    keys,
                    whole_worker_shutdown=authorization.stop_accepting_all_computation,
                )
            if not removed:
                if self._is_superseded(authorization):
                    self._remember_superseded(authorization)
                    return
                raise RuntimeError("Controller drain uses an obsolete placement generation")
            report_sequence = await self._reporter.flush()
            await self._wait_report_enqueued(report_sequence)
            self._remember_completed(authorization)
            if authorization.stop_accepting_all_computation:
                self._shutdown_drained.set()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._drain_error is None:
                self._drain_error = error
                self._drain_failed.set()
        finally:
            active = self._active_drains.get(authorization.drain_id)
            if active is not None and active.task is asyncio.current_task():
                self._active_drains.pop(authorization.drain_id, None)

    def _is_superseded(self, authorization: DrainAuthorization) -> bool:
        return (
            not authorization.stop_accepting_all_computation
            and authorization.placement_generation < self._current_placement_generation
        )

    def _remember_superseded(self, authorization: DrainAuthorization) -> None:
        previous = self._superseded_drains.get(authorization.drain_id)
        if previous is not None and previous != authorization:
            raise ValueError("superseded drain ID was repeated with different content")
        if previous is None and len(self._superseded_drains) >= _COMPLETED_DRAIN_HISTORY:
            self._superseded_drains.popitem(last=False)
        self._superseded_drains[authorization.drain_id] = authorization

    def _remember_completed(self, authorization: DrainAuthorization) -> None:
        if len(self._completed_drains) >= _COMPLETED_DRAIN_HISTORY:
            self._completed_drains.popitem(last=False)
        self._completed_drains[authorization.drain_id] = authorization
        self._completion_changed.set()

    async def _wait_drain_failure(self) -> Exception:
        await self._drain_failed.wait()
        error = self._drain_error
        if error is None:
            raise RuntimeError("drain failure event has no error")
        return error
