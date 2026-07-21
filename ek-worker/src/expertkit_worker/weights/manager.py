"""Controller-driven expert loading, placement, and ready-state ownership."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import structlog

from expertkit_worker.weights.adapter import (
    WeightAdapter,
    WeightPlacementFatalError,
)
from expertkit_worker.weights.dram_cache import WeightKey
from expertkit_worker.weights.loader import (
    CpuWeightLease,
    CpuWeightLoader,
    WeightLoadErrorCode,
    WeightLoadFailed,
    WeightLoadFailure,
    WeightLoadStage,
    WeightSource,
)
from expertkit_worker.weights.ready import ReadyWeightLease, ReadyWeightTable
from expertkit_worker.weights.writeback import DiskWriteback

logger = structlog.get_logger(__name__)


class ExpertStateKind(StrEnum):
    """States reported to the Controller for one expert."""

    READY = "ready"
    FAILED = "failed"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class TargetExpert:
    """Describe one Controller-requested expert on this Worker's fixed device."""

    layer_id: int
    expert_id: int
    target_device: str
    peer_endpoints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "peer_endpoints", tuple(self.peer_endpoints))
        for name, value in (("layer_id", self.layer_id), ("expert_id", self.expert_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.target_device:
            raise ValueError("target_device must not be empty")
        if any(not endpoint for endpoint in self.peer_endpoints):
            raise ValueError("peer endpoints must not be empty")

    @property
    def key(self) -> WeightKey:
        """Return the direct-table position for this target."""

        return WeightKey(self.layer_id, self.expert_id)


@dataclass(frozen=True, slots=True)
class ExpertState:
    """Describe the latest reportable result for one expert."""

    key: WeightKey
    state: ExpertStateKind
    failure: WeightLoadFailure | None = None

    def __post_init__(self) -> None:
        if self.state is ExpertStateKind.FAILED and self.failure is None:
            raise ValueError("FAILED expert state requires failure details")
        if self.state is not ExpertStateKind.FAILED and self.failure is not None:
            raise ValueError("only FAILED expert state may include failure details")


@dataclass(frozen=True, slots=True)
class ExpertStateChange:
    """Associate one expert result with the latest placement generation."""

    placement_generation: int
    expert: ExpertState


@dataclass(frozen=True, slots=True)
class WeightManagerStats:
    """Report current assigned, loading, ready, and device-memory counts."""

    placement_generation: int
    assigned_experts: int
    loading_experts: int
    ready_experts: int
    loaded_device_bytes: int
    device_weight_capacity_bytes: int
    max_experts: int


@dataclass(frozen=True, slots=True)
class _RemovalRequest:
    placement_generation: int
    keys: tuple[WeightKey, ...]
    whole_worker_shutdown: bool


class WeightManagerFatalError(RuntimeError):
    """Associate one fatal placement failure with its expert position."""

    def __init__(self, key: WeightKey, error: WeightPlacementFatalError) -> None:
        super().__init__(str(error))
        self.key = key
        self.reason = error.reason
        self.diagnostic = error.diagnostic


class WeightManager[CpuWeightT, ReadyWeightT]:
    """Apply complete placement generations and own the only ready-weight table."""

    def __init__(
        self,
        *,
        num_layers: int,
        experts_per_layer: int,
        device: str,
        device_weight_capacity_bytes: int,
        max_concurrent_loads: int,
        adapter: WeightAdapter[CpuWeightT, ReadyWeightT],
        loader: CpuWeightLoader[CpuWeightT, ReadyWeightT],
        writeback: DiskWriteback[CpuWeightT],
        state_changed: Callable[[ExpertStateChange], None] | None = None,
        device_bytes_changed: Callable[[int], None] | None = None,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("device_weight_capacity_bytes", device_weight_capacity_bytes),
            ("max_concurrent_loads", max_concurrent_loads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not device:
            raise ValueError("device must not be empty")
        ready_weight_bytes = adapter.ready_weight_bytes()
        if ready_weight_bytes <= 0:
            raise ValueError("Weight adapter ready size must be positive")
        max_experts = device_weight_capacity_bytes // ready_weight_bytes
        if max_experts <= 0:
            raise ValueError("device weight capacity cannot fit one expert")
        adapter.initialize_ready_storage(max_experts)

        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._device = device
        self._device_weight_capacity_bytes = device_weight_capacity_bytes
        self._ready_weight_bytes = ready_weight_bytes
        self._max_experts = max_experts
        self._adapter = adapter
        self._loader = loader
        self._writeback = writeback
        self._state_changed = state_changed or (lambda _change: None)
        self._device_bytes_changed = device_bytes_changed or (lambda _byte_count: None)
        self._ready = ReadyWeightTable[ReadyWeightT](num_layers, experts_per_layer)
        self._load_limit = asyncio.Semaphore(max_concurrent_loads)
        self._conversion_executor = ThreadPoolExecutor(
            max_workers=max_concurrent_loads,
            thread_name_prefix="ek-weight-convert",
        )
        self._lock = asyncio.Lock()
        self._placement_generation = 0
        self._targets: dict[WeightKey, TargetExpert] = {}
        self._states: dict[WeightKey, ExpertState] = {}
        self._load_tasks: dict[WeightKey, asyncio.Task[None]] = {}
        self._removal_lock = asyncio.Lock()
        self._removal_request: _RemovalRequest | None = None
        self._removal_task: asyncio.Task[bool] | None = None
        self._ready_keys: set[WeightKey] = set()
        self._loaded_device_bytes = 0
        self._logged_placement_generation = 0
        self._load_started_at = 0.0
        self._load_progress_step = 1
        self._next_load_progress_count = 1
        self._load_completion_logged = False
        self._fatal_error: WeightManagerFatalError | None = None
        self._fatal_event = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None
        self._started = False
        self._shutting_down = False
        self._closed = False

    @property
    def max_experts(self) -> int:
        """Return equal-sized expert capacity reported during registration."""

        return self._max_experts

    @property
    def fatal_error(self) -> WeightManagerFatalError | None:
        """Return the first fatal placement error, if one occurred."""

        return self._fatal_error

    def start(self) -> None:
        """Start background writeback before accepting placement generations."""

        if self._closed:
            raise RuntimeError("Weight Manager is closed")
        if self._started:
            return
        self._started = True
        self._writeback.start()

    def acquire_many(
        self,
        layer_id: int,
        distinct_expert_ids: tuple[int, ...],
    ) -> ReadyWeightLease[ReadyWeightT]:
        """Retain ready Backend objects without loading, copying, or waiting."""

        return self._ready.acquire_many(layer_id, distinct_expert_ids)

    async def apply_targets(
        self,
        placement_generation: int,
        targets: Iterable[TargetExpert],
    ) -> bool:
        """Atomically apply one complete target list and start missing loads."""

        if not self._started:
            raise RuntimeError("Weight Manager has not been started")
        if isinstance(placement_generation, bool) or not isinstance(placement_generation, int):
            raise ValueError("placement_generation must be an integer")
        if placement_generation <= 0:
            raise ValueError("placement_generation must be positive")
        resolved = self._validate_targets(targets)
        changes: list[ExpertStateChange] = []
        placement_log: dict[str, int] | None = None
        async with self._lock:
            if self._closed:
                raise RuntimeError("Weight Manager is closed")
            if placement_generation < self._placement_generation:
                return False
            if placement_generation == self._placement_generation:
                if resolved != self._targets:
                    raise ValueError("one placement generation cannot contain different targets")
                return False
            if self._shutting_down and any(key not in self._targets for key in resolved):
                raise RuntimeError("shutting-down Weight Manager rejects new expert targets")

            previous_targets = self._targets
            self._placement_generation = placement_generation
            self._targets = resolved

            for key in previous_targets.keys() - resolved.keys():
                task = self._load_tasks.get(key)
                if task is not None:
                    task.cancel()
                if key not in self._ready_keys:
                    change = self._set_state_locked(
                        key,
                        ExpertState(key, ExpertStateKind.REMOVED),
                    )
                    if change is not None:
                        changes.append(change)

            for key in resolved:
                if key in self._ready_keys:
                    continue
                self._states.pop(key, None)
                if key not in self._load_tasks and not self._shutting_down:
                    self._start_load_locked(key)

            ready_count = len(self._ready_keys.intersection(resolved))
            assigned_count = len(resolved)
            loading_count = sum(key in self._load_tasks for key in resolved)
            self._logged_placement_generation = placement_generation
            self._load_started_at = time.monotonic()
            self._load_progress_step = max(1, (assigned_count + 9) // 10)
            self._next_load_progress_count = (
                (ready_count // self._load_progress_step) + 1
            ) * self._load_progress_step
            self._load_completion_logged = assigned_count > 0 and ready_count == assigned_count
            placement_log = {
                "placement_generation": placement_generation,
                "assigned_experts": assigned_count,
                "ready_experts": ready_count,
                "loading_experts": loading_count,
                "added_experts": len(resolved.keys() - previous_targets.keys()),
                "removed_experts": len(previous_targets.keys() - resolved.keys()),
            }

        self._emit(changes)
        assert placement_log is not None
        logger.info("expert_placement_applied", **placement_log)
        if placement_log["loading_experts"]:
            logger.info(
                "expert_loading_started",
                placement_generation=placement_generation,
                assigned_experts=placement_log["assigned_experts"],
                ready_experts=placement_log["ready_experts"],
                loading_experts=placement_log["loading_experts"],
            )
        elif (
            placement_log["assigned_experts"]
            and placement_log["ready_experts"] == placement_log["assigned_experts"]
        ):
            logger.info(
                "expert_loading_completed",
                placement_generation=placement_generation,
                ready_experts=placement_log["ready_experts"],
                duration_ms=0.0,
                loaded_device_bytes=self._loaded_device_bytes,
            )
        return True

    async def begin_shutdown(self) -> bool:
        """Stop current loading and reject targets not already assigned.

        Ready objects remain acquirable until the Controller authorizes their
        computation drain and removal.
        """

        async with self._lock:
            if self._closed:
                raise RuntimeError("Weight Manager is closed")
            if self._shutting_down:
                return False
            self._shutting_down = True
            for task in self._load_tasks.values():
                task.cancel()
            return True

    async def snapshot(self) -> tuple[int, tuple[ExpertState, ...]]:
        """Return one consistent complete state snapshot for stream recovery."""

        async with self._lock:
            return self._placement_generation, tuple(
                self._states[key] for key in sorted(self._states)
            )

    async def remove_after_drain(
        self,
        placement_generation: int,
        keys: Iterable[WeightKey],
        *,
        whole_worker_shutdown: bool = False,
    ) -> bool:
        """Withdraw drained ready objects after Transport admission is idle.

        Returns:
            ``False`` when the authorization belongs to an older placement
            generation. Repeating the current request is idempotent.

        Warning:
            The caller must first stop matching Transport admission and wait for
            waiting and active batches to drain. Existing Backend references are
            still waited here before an object becomes ``REMOVED``. Assigned
            experts may be removed only during an authorized whole-Worker shutdown.
        """

        if not isinstance(whole_worker_shutdown, bool):
            raise TypeError("whole_worker_shutdown must be a Boolean")

        request = _RemovalRequest(
            placement_generation=self._validate_placement_generation(placement_generation),
            keys=self._validate_weight_keys(keys),
            whole_worker_shutdown=whole_worker_shutdown,
        )
        while True:
            async with self._removal_lock:
                task = self._removal_task
                active_request = self._removal_request
                if task is None:
                    task = asyncio.create_task(
                        self._remove_after_drain(request),
                        name=f"weight-remove-generation-{placement_generation}",
                    )
                    self._removal_request = request
                    self._removal_task = task
                    active_request = request
            try:
                result = await asyncio.shield(task)
            finally:
                if task.done():
                    async with self._removal_lock:
                        if self._removal_task is task:
                            self._removal_task = None
                            self._removal_request = None
            if active_request == request:
                return result

    async def stats(self) -> WeightManagerStats:
        """Return consistent resource and lifecycle counts."""

        async with self._lock:
            return WeightManagerStats(
                placement_generation=self._placement_generation,
                assigned_experts=len(self._targets),
                loading_experts=len(self._load_tasks),
                ready_experts=len(self._ready_keys),
                loaded_device_bytes=self._loaded_device_bytes,
                device_weight_capacity_bytes=self._device_weight_capacity_bytes,
                max_experts=self._max_experts,
            )

    async def wait_for_idle(self) -> None:
        """Wait until every current placement load has completed or been cancelled."""

        while True:
            async with self._lock:
                tasks = tuple(self._load_tasks.values())
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0)

    async def wait_fatal(self) -> WeightManagerFatalError:
        """Wait for the first placement failure that requires process termination."""

        await self._fatal_event.wait()
        error = self._fatal_error
        if error is None:
            raise RuntimeError("fatal Weight Manager event has no error")
        return error

    async def close(self) -> None:
        """Cancel loading, drain writeback, and release every ready object."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close(),
                name="weight-manager-close",
            )
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        """Perform the single close sequence retained across caller cancellation."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._load_tasks.values())
            for task in tasks:
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._writeback.close()

        async with self._lock:
            ready_keys = tuple(sorted(self._ready_keys))
        loop = asyncio.get_running_loop()
        for key in ready_keys:
            self._ready.begin_withdrawal(key.layer_id, key.expert_id)
            await loop.run_in_executor(
                self._conversion_executor,
                partial(
                    self._finish_withdrawal_and_release,
                    key.layer_id,
                    key.expert_id,
                ),
            )
        self._conversion_executor.shutdown(wait=True, cancel_futures=True)
        async with self._lock:
            self._ready_keys.clear()
            self._loaded_device_bytes = 0
        self._emit_device_bytes(0)

    def _validate_targets(
        self,
        targets: Iterable[TargetExpert],
    ) -> dict[WeightKey, TargetExpert]:
        resolved: dict[WeightKey, TargetExpert] = {}
        for target in targets:
            if not isinstance(target, TargetExpert):
                raise TypeError("targets must contain TargetExpert values")
            key = target.key
            if key.layer_id >= self._num_layers or key.expert_id >= self._experts_per_layer:
                raise ValueError("target expert exceeds the configured model shape")
            if target.target_device != self._device:
                raise ValueError("target device does not match this Worker process")
            if key in resolved:
                raise ValueError("target expert list contains a duplicate")
            resolved[key] = target
        if len(resolved) > self._max_experts:
            raise ValueError("target expert count exceeds max_experts")
        return resolved

    @staticmethod
    def _validate_placement_generation(placement_generation: int) -> int:
        if isinstance(placement_generation, bool) or not isinstance(placement_generation, int):
            raise ValueError("placement_generation must be an integer")
        if placement_generation <= 0:
            raise ValueError("placement_generation must be positive")
        return placement_generation

    def _validate_weight_keys(self, keys: Iterable[WeightKey]) -> tuple[WeightKey, ...]:
        resolved = tuple(keys)
        if any(not isinstance(key, WeightKey) for key in resolved):
            raise TypeError("drained experts must be WeightKey values")
        if len(set(resolved)) != len(resolved):
            raise ValueError("drained experts must not contain duplicates")
        for key in resolved:
            if key.layer_id >= self._num_layers or key.expert_id >= self._experts_per_layer:
                raise ValueError("drained expert exceeds the configured model shape")
        return tuple(sorted(resolved))

    async def _remove_after_drain(self, request: _RemovalRequest) -> bool:
        changes: list[ExpertStateChange] = []
        async with self._lock:
            if self._closed:
                raise RuntimeError("Weight Manager is closed")
            if request.placement_generation < self._placement_generation:
                return False
            if request.placement_generation > self._placement_generation:
                raise ValueError("drain generation is newer than current placement")
            assigned = tuple(key for key in request.keys if key in self._targets)
            if assigned and not request.whole_worker_shutdown:
                raise ValueError("cannot remove an expert in the current target list")

            ready_keys = tuple(key for key in request.keys if key in self._ready_keys)
            for key in ready_keys:
                self._ready.begin_withdrawal(key.layer_id, key.expert_id)
            loop = asyncio.get_running_loop()
            for key in ready_keys:
                await loop.run_in_executor(
                    self._conversion_executor,
                    partial(
                        self._finish_withdrawal_and_release,
                        key.layer_id,
                        key.expert_id,
                    ),
                )
                self._ready_keys.remove(key)
                self._loaded_device_bytes -= self._ready_weight_bytes

            loaded_device_bytes = self._loaded_device_bytes

            for key in request.keys:
                change = self._set_state_locked(
                    key,
                    ExpertState(key, ExpertStateKind.REMOVED),
                )
                if change is not None:
                    changes.append(change)
        self._emit(changes)
        self._emit_device_bytes(loaded_device_bytes)
        return True

    def _start_load_locked(self, key: WeightKey) -> None:
        task = asyncio.create_task(
            self._load_one(key),
            name=f"weight-load-{key.layer_id}-{key.expert_id}",
        )
        self._load_tasks[key] = task

    async def _load_one(self, key: WeightKey) -> None:
        cancelled = False
        try:
            try:
                async with self._load_limit:
                    async with self._lock:
                        target = self._targets.get(key)
                    if target is None:
                        return
                    try:
                        cpu_lease = await self._loader.acquire(
                            key,
                            peer_endpoints=target.peer_endpoints,
                        )
                    except WeightLoadFailed as error:
                        await self._finish_failure(key, self._source_failure(error))
                        return

                    try:
                        try:
                            ready_weight, conversion_cancelled = await self._make_ready_weight(
                                key,
                                cpu_lease,
                            )
                            cancelled = cancelled or conversion_cancelled
                        except WeightPlacementFatalError as error:
                            await self._set_fatal(key, error)
                            return
                        except (ValueError, TypeError) as error:
                            await self._finish_failure(
                                key,
                                WeightLoadFailure(
                                    cpu_lease.source,
                                    WeightLoadStage.CONVERT,
                                    WeightLoadErrorCode.UNSUPPORTED,
                                    False,
                                    str(error),
                                ),
                            )
                            return
                        except Exception as error:
                            await self._finish_failure(
                                key,
                                WeightLoadFailure(
                                    cpu_lease.source,
                                    WeightLoadStage.CONVERT,
                                    WeightLoadErrorCode.INTERNAL,
                                    False,
                                    str(error),
                                ),
                            )
                            return
                        if cancelled:
                            self._adapter.release_ready_weight(ready_weight)
                        else:
                            cancelled = await self._finish_ready_without_leaking_on_cancel(
                                key,
                                ready_weight,
                            )
                    finally:
                        if cpu_lease.source in {
                            WeightSource.PEER,
                            WeightSource.WEIGHT_SERVER,
                        } and self._writeback.try_submit(key, cpu_lease):
                            pass
                        else:
                            await cpu_lease.close()
            except asyncio.CancelledError:
                cancelled = True
        finally:
            await self._load_ended(key, cancelled)

    async def _make_ready_weight(
        self,
        key: WeightKey,
        lease: CpuWeightLease[CpuWeightT],
    ) -> tuple[ReadyWeightT, bool]:
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(
            self._conversion_executor,
            partial(
                self._adapter.make_ready_weight,
                lease.cached.value,
                layer_id=key.layer_id,
                expert_id=key.expert_id,
            ),
        )
        cancelled = False
        while True:
            try:
                return await asyncio.shield(work), cancelled
            except asyncio.CancelledError:
                cancelled = True

    async def _finish_ready(self, key: WeightKey, ready_weight: ReadyWeightT) -> None:
        changes: list[ExpertStateChange] = []
        loaded_device_bytes: int | None = None
        progress_log: dict[str, int | float] | None = None
        completed_log: dict[str, int | float] | None = None
        published = False
        try:
            async with self._lock:
                if self._closed or key not in self._targets or key in self._ready_keys:
                    return
                next_loaded_bytes = self._loaded_device_bytes + self._ready_weight_bytes
                if next_loaded_bytes > self._device_weight_capacity_bytes:
                    failure = WeightLoadFailure(
                        WeightSource.DRAM,
                        WeightLoadStage.PLACE,
                        WeightLoadErrorCode.INTERNAL,
                        False,
                        "ready expert exceeds the derived device weight capacity",
                    )
                    change = self._set_state_locked(
                        key,
                        ExpertState(key, ExpertStateKind.FAILED, failure),
                    )
                    if change is not None:
                        changes.append(change)
                else:
                    self._ready.publish(key.layer_id, key.expert_id, ready_weight)
                    published = True
                    self._ready_keys.add(key)
                    self._loaded_device_bytes += self._ready_weight_bytes
                    loaded_device_bytes = self._loaded_device_bytes
                    change = self._set_state_locked(
                        key,
                        ExpertState(key, ExpertStateKind.READY),
                    )
                    if change is not None:
                        changes.append(change)
                    if self._logged_placement_generation == self._placement_generation:
                        assigned_count = len(self._targets)
                        ready_count = len(self._ready_keys.intersection(self._targets))
                        remaining_count = assigned_count - ready_count
                        elapsed_ms = (time.monotonic() - self._load_started_at) * 1000
                        if (
                            assigned_count > 0
                            and ready_count == assigned_count
                            and not self._load_completion_logged
                        ):
                            self._load_completion_logged = True
                            completed_log = {
                                "placement_generation": self._placement_generation,
                                "ready_experts": ready_count,
                                "duration_ms": elapsed_ms,
                                "loaded_device_bytes": self._loaded_device_bytes,
                            }
                        elif ready_count >= self._next_load_progress_count:
                            while self._next_load_progress_count <= ready_count:
                                self._next_load_progress_count += self._load_progress_step
                            progress_log = {
                                "placement_generation": self._placement_generation,
                                "ready_experts": ready_count,
                                "assigned_experts": assigned_count,
                                "remaining_experts": remaining_count,
                                "progress_percent": ready_count * 100.0 / assigned_count,
                                "duration_ms": elapsed_ms,
                            }
        finally:
            if not published:
                self._adapter.release_ready_weight(ready_weight)
        self._emit(changes)
        if loaded_device_bytes is not None:
            self._emit_device_bytes(loaded_device_bytes)
        if progress_log is not None:
            logger.info("expert_loading_progress", **progress_log)
        if completed_log is not None:
            logger.info("expert_loading_completed", **completed_log)

    async def _finish_ready_without_leaking_on_cancel(
        self,
        key: WeightKey,
        ready_weight: ReadyWeightT,
    ) -> bool:
        """Finish the ready transition even if target replacement cancels its load."""

        work = asyncio.create_task(self._finish_ready(key, ready_weight))
        cancelled = False
        while True:
            try:
                await asyncio.shield(work)
                return cancelled
            except asyncio.CancelledError:
                cancelled = True

    def _finish_withdrawal_and_release(self, layer_id: int, expert_id: int) -> None:
        ready_weight = self._ready.finish_withdrawal(layer_id, expert_id)
        self._adapter.release_ready_weight(ready_weight)

    async def _finish_failure(self, key: WeightKey, failure: WeightLoadFailure) -> None:
        changes: list[ExpertStateChange] = []
        changed = False
        async with self._lock:
            if self._closed or key not in self._targets or key in self._ready_keys:
                return
            change = self._set_state_locked(
                key,
                ExpertState(key, ExpertStateKind.FAILED, failure),
            )
            if change is not None:
                changes.append(change)
                changed = True
        self._emit(changes)
        if changed:
            logger.error(
                "expert_loading_failed",
                layer_id=key.layer_id,
                expert_id=key.expert_id,
                placement_generation=self._placement_generation,
                source=failure.source.value,
                stage=failure.stage.value,
                error_code=failure.code.value,
                retryable=failure.retryable,
                diagnostic=failure.diagnostic,
            )

    async def _set_fatal(self, key: WeightKey, error: WeightPlacementFatalError) -> None:
        async with self._lock:
            if self._fatal_error is None:
                self._fatal_error = WeightManagerFatalError(key, error)
                self._fatal_event.set()

    async def _load_ended(self, key: WeightKey, cancelled: bool) -> None:
        async with self._lock:
            current = asyncio.current_task()
            if self._load_tasks.get(key) is current:
                self._load_tasks.pop(key)
            should_restart = (
                cancelled
                and not self._closed
                and not self._shutting_down
                and key in self._targets
                and key not in self._ready_keys
            )
            if should_restart:
                self._states.pop(key, None)
                self._start_load_locked(key)

    def _set_state_locked(
        self,
        key: WeightKey,
        state: ExpertState,
    ) -> ExpertStateChange | None:
        if self._states.get(key) == state:
            return None
        self._states[key] = state
        return ExpertStateChange(self._placement_generation, state)

    @staticmethod
    def _source_failure(error: WeightLoadFailed) -> WeightLoadFailure:
        last = error.failures[-1]
        return WeightLoadFailure(
            last.source,
            last.stage,
            last.code,
            error.retryable,
            str(error),
        )

    def _emit(self, changes: Iterable[ExpertStateChange]) -> None:
        for change in changes:
            try:
                self._state_changed(change)
            except Exception:
                logger.error(
                    "weight_state_callback_failed",
                    layer_id=change.expert.key.layer_id,
                    expert_id=change.expert.key.expert_id,
                    placement_generation=change.placement_generation,
                    exc_info=True,
                )

    def _emit_device_bytes(self, byte_count: int) -> None:
        try:
            self._device_bytes_changed(byte_count)
        except Exception:
            logger.error(
                "device_weight_bytes_callback_failed",
                byte_count=byte_count,
                exc_info=True,
            )
