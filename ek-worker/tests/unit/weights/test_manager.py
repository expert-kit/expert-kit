"""Tests for Controller-driven expert placement and ready-weight ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass
from typing import Any

import pytest

from expertkit_worker.weights import (
    ExpertState,
    ExpertStateChange,
    ExpertStateKind,
    TargetExpert,
    WeightAdapter,
    WeightLoadErrorCode,
    WeightLoadFailed,
    WeightLoadFailure,
    WeightLoadStage,
    WeightManager,
    WeightManagerFatalError,
    WeightPlacementFatalError,
    WeightPlacementFatalReason,
    WeightsNotReady,
    WeightSource,
)
from expertkit_worker.weights.dram_cache import WeightKey


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run one isolated asynchronous placement scenario."""

    return asyncio.run(coroutine)


async def _await_with_loop_yields[T](awaitable: Awaitable[T]) -> T:
    """Keep the test loop runnable while an executor thread completes work."""

    task = asyncio.ensure_future(awaitable)
    async with asyncio.timeout(2):
        while not task.done():
            await asyncio.sleep(0)
    return task.result()


@dataclass(frozen=True, slots=True)
class _CachedWeight:
    value: object


class _FakeLease:
    def __init__(self, value: object, source: WeightSource = WeightSource.DRAM) -> None:
        self.cached = _CachedWeight(value)
        self.source = source
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeLoader:
    def __init__(self) -> None:
        self.calls: list[tuple[WeightKey, tuple[str, ...]]] = []
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.failures: dict[WeightKey, WeightLoadFailed] = {}
        self.values: dict[WeightKey, object] = {}

    async def acquire(
        self,
        key: WeightKey,
        *,
        peer_endpoints: tuple[str, ...] = (),
    ) -> _FakeLease:
        self.calls.append((key, peer_endpoints))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.block:
                await self.release.wait()
            failure = self.failures.get(key)
            if failure is not None:
                raise failure
            return _FakeLease(self.values.get(key, key))
        finally:
            self.active -= 1


class _FakeWriteback:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def try_submit(self, _key: WeightKey, _lease: object) -> bool:
        return False

    async def close(self) -> None:
        self.closed = True


class _FakeAdapter(WeightAdapter[object, object]):
    def __init__(self) -> None:
        self.initialized_capacities: list[int] = []
        self.released: list[object] = []

    @property
    def backend_name(self) -> str:
        return "fake"

    def make_cpu_weight(self, source: object) -> object:
        return source

    def make_ready_weight(
        self,
        cpu_weight: object,
        *,
        layer_id: int,
        expert_id: int,
    ) -> object:
        del layer_id, expert_id
        if cpu_weight == "invalid":
            raise ValueError("unsupported weight metadata")
        if cpu_weight == "fatal":
            raise WeightPlacementFatalError(
                WeightPlacementFatalReason.DEVICE_OOM,
                "out of device memory",
            )
        return ("ready", cpu_weight)

    def cpu_extra_bytes(self) -> int:
        return 0

    def source_tensor_bytes(self) -> int:
        return 8

    def ready_weight_bytes(self) -> int:
        return 16

    def conversion_temporary_bytes(self) -> int:
        return 0

    def initialize_ready_storage(self, max_experts: int) -> None:
        self.initialized_capacities.append(max_experts)

    def release_ready_weight(self, ready_weight: object) -> None:
        self.released.append(ready_weight)


def _make_manager(
    loader: _FakeLoader,
    *,
    max_concurrent_loads: int = 2,
    device_weight_capacity_bytes: int = 64,
    changes: list[ExpertStateChange] | None = None,
    byte_changes: list[int] | None = None,
    adapter: _FakeAdapter | None = None,
) -> tuple[WeightManager[object, object], _FakeWriteback]:
    writeback = _FakeWriteback()
    selected_adapter = adapter or _FakeAdapter()
    manager = WeightManager(
        num_layers=2,
        experts_per_layer=4,
        device="cuda:0",
        device_weight_capacity_bytes=device_weight_capacity_bytes,
        max_concurrent_loads=max_concurrent_loads,
        adapter=selected_adapter,
        loader=loader,
        writeback=writeback,
        state_changed=None if changes is None else changes.append,
        device_bytes_changed=None if byte_changes is None else byte_changes.append,
    )
    return manager, writeback


def test_manager_initializes_and_releases_backend_ready_storage() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        adapter = _FakeAdapter()
        manager, _ = _make_manager(loader, adapter=adapter)
        manager.start()

        assert adapter.initialized_capacities == [4]
        await manager.apply_targets(1, [_target(0, 0), _target(1, 1)])
        await _await_with_loop_yields(manager.wait_for_idle())
        retained = manager.acquire_many(0, (0,))
        await manager.apply_targets(2, [_target(1, 1)])

        removal = asyncio.create_task(manager.remove_after_drain(2, (WeightKey(0, 0),)))
        await asyncio.sleep(0)
        assert adapter.released == []

        retained.close()
        assert await _await_with_loop_yields(removal) is True
        assert adapter.released == [("ready", WeightKey(0, 0))]

        await _await_with_loop_yields(manager.close())
        assert adapter.released == [
            ("ready", WeightKey(0, 0)),
            ("ready", WeightKey(1, 1)),
        ]

    run(scenario())


def _target(layer_id: int, expert_id: int, *peers: str) -> TargetExpert:
    return TargetExpert(layer_id, expert_id, "cuda:0", peers)


def test_manager_bounds_whole_load_pipeline_and_publishes_direct_references() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        loader.block = True
        changes: list[ExpertStateChange] = []
        byte_changes: list[int] = []
        manager, writeback = _make_manager(
            loader,
            max_concurrent_loads=2,
            changes=changes,
            byte_changes=byte_changes,
        )
        manager.start()

        await manager.apply_targets(1, [_target(0, 0), _target(0, 1), _target(0, 2)])
        await asyncio.wait_for(loader.started.wait(), timeout=1)
        async with asyncio.timeout(1):
            while loader.active < 2:
                await asyncio.sleep(0)
        assert loader.max_active == 2
        loader.release.set()
        await _await_with_loop_yields(manager.wait_for_idle())

        lease = manager.acquire_many(0, (0, 1, 2))
        assert lease.objects == (
            ("ready", WeightKey(0, 0)),
            ("ready", WeightKey(0, 1)),
            ("ready", WeightKey(0, 2)),
        )
        assert [change.expert.state for change in changes] == [
            ExpertStateKind.READY,
            ExpertStateKind.READY,
            ExpertStateKind.READY,
        ]
        stats = await manager.stats()
        assert stats.ready_experts == 3
        assert stats.loaded_device_bytes == 48
        assert stats.max_experts == 4
        lease.close()
        await _await_with_loop_yields(manager.close())
        assert byte_changes == [16, 32, 48, 0]
        assert writeback.started is True
        assert writeback.closed is True

    run(scenario())


def test_new_generation_reuses_inflight_load_for_same_expert() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        loader.block = True
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()

        await manager.apply_targets(1, [_target(0, 1, "http://old-peer")])
        await asyncio.wait_for(loader.started.wait(), timeout=1)
        await manager.apply_targets(2, [_target(0, 1, "http://new-peer")])
        loader.release.set()
        await _await_with_loop_yields(manager.wait_for_idle())

        assert loader.calls == [(WeightKey(0, 1), ("http://old-peer",))]
        assert len(changes) == 1
        assert changes[0].placement_generation == 2
        assert changes[0].expert.state is ExpertStateKind.READY
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_removed_never_ready_expert_is_cancelled_and_reported() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        loader.block = True
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()

        await manager.apply_targets(1, [_target(1, 3)])
        await asyncio.wait_for(loader.started.wait(), timeout=1)
        await manager.apply_targets(2, [])
        await _await_with_loop_yields(manager.wait_for_idle())

        assert len(changes) == 1
        assert changes[0].placement_generation == 2
        assert changes[0].expert.key == WeightKey(1, 3)
        assert changes[0].expert.state is ExpertStateKind.REMOVED
        stats = await manager.stats()
        assert stats.loading_experts == 0
        assert stats.ready_experts == 0
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_shutdown_stops_loading_and_rejects_new_expert_targets() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        loader.block = True
        manager, _ = _make_manager(loader)
        manager.start()

        await manager.apply_targets(1, [_target(0, 0)])
        await asyncio.wait_for(loader.started.wait(), timeout=1)
        assert await manager.begin_shutdown() is True
        assert await manager.begin_shutdown() is False
        await _await_with_loop_yields(manager.wait_for_idle())

        assert await manager.apply_targets(2, [_target(0, 0)]) is True
        assert len(loader.calls) == 1
        with pytest.raises(RuntimeError, match="rejects new expert targets"):
            await manager.apply_targets(3, [_target(0, 1)])
        assert await manager.apply_targets(3, []) is True

        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_source_and_conversion_failures_are_isolated_per_expert() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        source_key = WeightKey(0, 0)
        invalid_key = WeightKey(0, 1)
        ready_key = WeightKey(0, 2)
        loader.failures[source_key] = WeightLoadFailed(
            source_key,
            (
                WeightLoadFailure(
                    WeightSource.WEIGHT_SERVER,
                    WeightLoadStage.FETCH,
                    WeightLoadErrorCode.NETWORK,
                    True,
                    "temporarily unavailable",
                ),
            ),
        )
        loader.values[invalid_key] = "invalid"
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()

        await manager.apply_targets(
            1,
            [_target(0, 0), _target(0, 1), _target(0, 2)],
        )
        await _await_with_loop_yields(manager.wait_for_idle())

        states = {change.expert.key: change.expert for change in changes}
        assert states[source_key].failure is not None
        assert states[source_key].failure.retryable is True
        assert states[invalid_key].failure is not None
        assert states[invalid_key].failure.code is WeightLoadErrorCode.UNSUPPORTED
        assert states[ready_key].state is ExpertStateKind.READY
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_fatal_device_placement_error_requires_process_termination() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        key = WeightKey(1, 2)
        loader.values[key] = "fatal"
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()

        await manager.apply_targets(1, [_target(1, 2)])
        fatal = await _await_with_loop_yields(manager.wait_fatal())
        await _await_with_loop_yields(manager.wait_for_idle())

        assert isinstance(fatal, WeightManagerFatalError)
        assert fatal.key == key
        assert fatal.reason is WeightPlacementFatalReason.DEVICE_OOM
        assert changes == []
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_manager_rejects_invalid_generation_and_capacity_before_loading() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        manager, _ = _make_manager(
            loader,
            device_weight_capacity_bytes=32,
        )
        manager.start()

        assert await manager.apply_targets(2, [_target(0, 0)]) is True
        assert await manager.apply_targets(1, [_target(0, 1)]) is False
        assert await manager.apply_targets(2, [_target(0, 0)]) is False
        with pytest.raises(ValueError, match="different targets"):
            await manager.apply_targets(2, [_target(0, 1)])
        with pytest.raises(ValueError, match="exceeds max_experts"):
            await manager.apply_targets(
                3,
                [_target(0, 0), _target(0, 1), _target(0, 2)],
            )

        await _await_with_loop_yields(manager.wait_for_idle())
        assert len(loader.calls) == 1
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_remove_after_drain_waits_for_backend_reference_before_removed() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()
        key = WeightKey(0, 2)
        await manager.apply_targets(1, [_target(0, 2)])
        await _await_with_loop_yields(manager.wait_for_idle())
        lease = manager.acquire_many(0, (2,))
        await manager.apply_targets(2, [])

        removal = asyncio.create_task(manager.remove_after_drain(2, (key,)))
        async with asyncio.timeout(1):
            while True:
                try:
                    probe = manager.acquire_many(0, (2,))
                except WeightsNotReady:
                    break
                probe.close()
                await asyncio.sleep(0)
        assert removal.done() is False
        with pytest.raises(WeightsNotReady):
            manager.acquire_many(0, (2,))

        lease.close()
        assert await _await_with_loop_yields(removal) is True
        assert changes[-1].placement_generation == 2
        assert changes[-1].expert.key == key
        assert changes[-1].expert.state is ExpertStateKind.REMOVED
        stats = await manager.stats()
        assert stats.ready_experts == 0
        assert stats.loaded_device_bytes == 0
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_remove_after_drain_survives_caller_cancellation_and_is_idempotent() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()
        key = WeightKey(1, 1)
        await manager.apply_targets(1, [_target(1, 1)])
        await _await_with_loop_yields(manager.wait_for_idle())
        lease = manager.acquire_many(1, (1,))
        await manager.apply_targets(2, [])

        cancelled = asyncio.create_task(manager.remove_after_drain(2, (key,)))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        lease.close()

        assert await _await_with_loop_yields(manager.remove_after_drain(2, (key,))) is True
        removed = [
            change
            for change in changes
            if change.expert.key == key and change.expert.state is ExpertStateKind.REMOVED
        ]
        assert len(removed) == 1
        assert removed[0].placement_generation == 2
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_remove_after_drain_validates_generation_and_current_targets() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        manager, _ = _make_manager(loader)
        manager.start()
        key = WeightKey(0, 0)
        await manager.apply_targets(2, [_target(0, 0)])

        assert await manager.remove_after_drain(1, (key,)) is False
        with pytest.raises(ValueError, match="newer than current"):
            await manager.remove_after_drain(3, (key,))
        with pytest.raises(ValueError, match="current target"):
            await manager.remove_after_drain(2, (key,))

        await _await_with_loop_yields(manager.wait_for_idle())
        await _await_with_loop_yields(manager.close())

    run(scenario())


def test_whole_worker_shutdown_removes_current_targets_after_drain() -> None:
    async def scenario() -> None:
        loader = _FakeLoader()
        changes: list[ExpertStateChange] = []
        manager, _ = _make_manager(loader, changes=changes)
        manager.start()
        key = WeightKey(0, 0)
        await manager.apply_targets(1, [_target(0, 0)])
        await _await_with_loop_yields(manager.wait_for_idle())

        assert await manager.begin_shutdown() is True
        assert await manager.remove_after_drain(
            1,
            (key,),
            whole_worker_shutdown=True,
        )
        with pytest.raises(WeightsNotReady):
            manager.acquire_many(0, (0,))
        assert changes[-1] == ExpertStateChange(
            1,
            ExpertState(key, ExpertStateKind.REMOVED),
        )

        await _await_with_loop_yields(manager.close())

    run(scenario())
