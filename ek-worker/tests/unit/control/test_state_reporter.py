"""Tests for bounded batched expert-state reporting."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest
from expertkit_proto.ek.control.v2 import weight_control_pb2

from expertkit_worker.control import ExpertStateReporter
from expertkit_worker.weights import (
    ExpertState,
    ExpertStateChange,
    ExpertStateKind,
    WeightLoadErrorCode,
    WeightLoadFailure,
    WeightLoadStage,
    WeightSource,
)
from expertkit_worker.weights.dram_cache import WeightKey


class FakeClock:
    """Expose deterministic monotonic time without sleeping."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _state(
    expert_id: int,
    kind: ExpertStateKind = ExpertStateKind.READY,
    *,
    failure: WeightLoadFailure | None = None,
) -> ExpertState:
    return ExpertState(WeightKey(0, expert_id), kind, failure)


def _change(
    generation: int,
    expert_id: int,
    kind: ExpertStateKind = ExpertStateKind.READY,
    *,
    failure: WeightLoadFailure | None = None,
) -> ExpertStateChange:
    return ExpertStateChange(generation, _state(expert_id, kind, failure=failure))


def _reporter(
    clock: FakeClock,
    *,
    max_updates: int = 2,
) -> ExpertStateReporter:
    return ExpertStateReporter(
        num_layers=2,
        experts_per_layer=130,
        max_updates=max_updates,
        max_delay_ms=50,
        clock=clock,
    )


def test_reporter_coalesces_latest_state_and_flushes_at_size_limit() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        reporter = _reporter(clock)
        reporter.record(_change(4, 1, ExpertStateKind.FAILED, failure=_failure("old")))
        reporter.record(_change(4, 1))
        reporter.record(_change(4, 2))

        message = await reporter.take_updates()

        assert message is not None
        assert message.WhichOneof("message") == "state_updates"
        assert message.state_updates.placement_generation == 4
        assert message.state_updates.report_sequence == 1
        assert [state.expert_id for state in message.state_updates.experts] == [1, 2]
        assert [state.state for state in message.state_updates.experts] == [
            weight_control_pb2.EXPERT_READY,
            weight_control_pb2.EXPERT_READY,
        ]
        assert await reporter.take_updates(force=True) is None

    run(scenario())


def test_reporter_flushes_oldest_update_after_configured_delay() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        reporter = _reporter(clock, max_updates=4)
        reporter.record(_change(3, 7))
        clock.now = 0.040
        reporter.record(_change(3, 7, ExpertStateKind.REMOVED))
        clock.now = 0.050

        message = await reporter.take_updates()

        assert message is not None
        assert [state.expert_id for state in message.state_updates.experts] == [7]
        assert message.state_updates.experts[0].state == weight_control_pb2.EXPERT_REMOVED

    run(scenario())


def test_explicit_flush_wakes_a_waiting_report_and_returns_its_sequence() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        reporter = _reporter(clock, max_updates=64)
        reporter.record(_change(3, 7))
        waiting = asyncio.create_task(reporter.take_updates())

        sequence = await reporter.flush()
        message = await waiting

        assert message is not None
        assert message.state_updates.report_sequence == sequence == 1
        assert [state.expert_id for state in message.state_updates.experts] == [7]

    run(scenario())


def test_reporter_never_mixes_placement_generations() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        reporter = _reporter(clock, max_updates=4)
        reporter.record(_change(2, 1))
        reporter.record(_change(3, 2))

        first = await reporter.take_updates(force=True)
        second = await reporter.take_updates(force=True)

        assert first is not None
        assert second is not None
        assert first.state_updates.placement_generation == 2
        assert second.state_updates.placement_generation == 3
        assert first.state_updates.report_sequence == 1
        assert second.state_updates.report_sequence == 2

    run(scenario())


def test_full_state_is_chunked_and_retains_changes_after_snapshot_boundary() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        reporter = _reporter(clock)
        reporter.record(_change(5, 0))
        boundary = reporter.mark_changes()
        reporter.record(_change(5, 129, ExpertStateKind.REMOVED))
        states = tuple(_state(expert_id) for expert_id in range(130))

        parts = reporter.full_state_parts(
            5,
            states,
            discard_pending_through=boundary,
        )

        assert len(parts) == 3
        assert [part.full_state.part_index for part in parts] == [0, 1, 2]
        assert all(part.full_state.part_count == 3 for part in parts)
        assert all(part.full_state.report_sequence == 1 for part in parts)
        assert [len(part.full_state.experts) for part in parts] == [64, 64, 2]
        incremental = await reporter.take_updates(force=True)
        assert incremental is not None
        assert incremental.state_updates.report_sequence == 2
        assert [state.expert_id for state in incremental.state_updates.experts] == [129]
        assert incremental.state_updates.experts[0].state == weight_control_pb2.EXPERT_REMOVED

    run(scenario())


def test_failure_encoding_is_stable_and_bounds_utf8_diagnostic() -> None:
    async def scenario() -> None:
        clock = FakeClock()
        reporter = _reporter(clock)
        diagnostic = "界" * 1000
        reporter.record(
            _change(
                8,
                4,
                ExpertStateKind.FAILED,
                failure=_failure(diagnostic),
            )
        )

        message = await reporter.take_updates(force=True)

        assert message is not None
        wire = message.state_updates.experts[0]
        assert wire.state == weight_control_pb2.EXPERT_FAILED
        assert wire.failure.stage == weight_control_pb2.WEIGHT_LOAD_VALIDATE
        assert wire.failure.code == weight_control_pb2.WEIGHT_LOAD_ERROR_UNEXPECTED_METADATA
        assert wire.failure.retryable is False
        assert len(wire.failure.diagnostic.encode("utf-8")) <= 1024

    run(scenario())


def test_report_acknowledgements_are_monotonic_and_idempotent() -> None:
    clock = FakeClock()
    reporter = _reporter(clock)
    reporter.full_state_parts(1, (), discard_pending_through=0)

    assert reporter.acknowledge(1) is True
    assert reporter.acknowledge(1) is False
    with pytest.raises(ValueError, match="unsent"):
        reporter.acknowledge(2)

    reporter.record(_change(1, 0))
    run(reporter.take_updates(force=True))
    assert reporter.acknowledge(2) is True
    with pytest.raises(ValueError, match="backwards"):
        reporter.acknowledge(1)


def _failure(diagnostic: str) -> WeightLoadFailure:
    return WeightLoadFailure(
        source=WeightSource.WEIGHT_SERVER,
        stage=WeightLoadStage.VALIDATE,
        code=WeightLoadErrorCode.UNEXPECTED_METADATA,
        retryable=False,
        diagnostic=diagnostic,
    )
