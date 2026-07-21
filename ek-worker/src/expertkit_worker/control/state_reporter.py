"""Bounded coalescing and protobuf encoding of expert state reports."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from expertkit_proto.ek.control.v2 import weight_control_pb2

from expertkit_worker.control.parts import MAX_EXPERTS_PER_CONTROL_PART
from expertkit_worker.weights import (
    ExpertState,
    ExpertStateChange,
    ExpertStateKind,
    WeightLoadErrorCode,
    WeightLoadStage,
)
from expertkit_worker.weights.dram_cache import WeightKey

MAX_WEIGHT_DIAGNOSTIC_BYTES = 1024
_UINT64_MAX = (1 << 64) - 1
_STATE_KIND_TO_PROTO = {
    ExpertStateKind.READY: weight_control_pb2.EXPERT_READY,
    ExpertStateKind.FAILED: weight_control_pb2.EXPERT_FAILED,
    ExpertStateKind.REMOVED: weight_control_pb2.EXPERT_REMOVED,
}
_LOAD_STAGE_TO_PROTO = {
    WeightLoadStage.FETCH: weight_control_pb2.WEIGHT_LOAD_FETCH,
    WeightLoadStage.READ: weight_control_pb2.WEIGHT_LOAD_READ,
    WeightLoadStage.PARSE: weight_control_pb2.WEIGHT_LOAD_PARSE,
    WeightLoadStage.VALIDATE: weight_control_pb2.WEIGHT_LOAD_VALIDATE,
    WeightLoadStage.CONVERT: weight_control_pb2.WEIGHT_LOAD_CONVERT,
    WeightLoadStage.PLACE: weight_control_pb2.WEIGHT_LOAD_PLACE,
}
_LOAD_CODE_TO_PROTO = {
    WeightLoadErrorCode.NOT_FOUND: weight_control_pb2.WEIGHT_LOAD_ERROR_NOT_FOUND,
    WeightLoadErrorCode.IO: weight_control_pb2.WEIGHT_LOAD_ERROR_IO,
    WeightLoadErrorCode.NETWORK: weight_control_pb2.WEIGHT_LOAD_ERROR_NETWORK,
    WeightLoadErrorCode.INVALID_FORMAT: weight_control_pb2.WEIGHT_LOAD_ERROR_INVALID_FORMAT,
    WeightLoadErrorCode.UNEXPECTED_METADATA: (
        weight_control_pb2.WEIGHT_LOAD_ERROR_UNEXPECTED_METADATA
    ),
    WeightLoadErrorCode.UNSUPPORTED: weight_control_pb2.WEIGHT_LOAD_ERROR_UNSUPPORTED,
    WeightLoadErrorCode.INTERNAL: weight_control_pb2.WEIGHT_LOAD_ERROR_INTERNAL,
}


@dataclass(frozen=True, slots=True)
class _PendingState:
    change_index: int
    recorded_at: float
    change: ExpertStateChange


def _bounded_diagnostic(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_WEIGHT_DIAGNOSTIC_BYTES:
        return value
    return encoded[:MAX_WEIGHT_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")


def _validate_positive_uint64(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _UINT64_MAX:
        raise ValueError(f"{name} must be a positive uint64")


def _encode_state(state: ExpertState) -> weight_control_pb2.ExpertState:
    result = weight_control_pb2.ExpertState(
        layer_id=state.key.layer_id,
        expert_id=state.key.expert_id,
        state=_STATE_KIND_TO_PROTO[state.state],
    )
    failure = state.failure
    if failure is not None:
        result.failure.CopyFrom(
            weight_control_pb2.WeightLoadFailure(
                stage=_LOAD_STAGE_TO_PROTO[failure.stage],
                code=_LOAD_CODE_TO_PROTO[failure.code],
                retryable=failure.retryable,
                diagnostic=_bounded_diagnostic(failure.diagnostic),
            )
        )
    return result


class ExpertStateReporter:
    """Keep at most one pending state per model expert and batch reports."""

    def __init__(
        self,
        *,
        num_layers: int,
        experts_per_layer: int,
        max_updates: int,
        max_delay_ms: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
            ("max_updates", max_updates),
            ("max_delay_ms", max_delay_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_updates > MAX_EXPERTS_PER_CONTROL_PART:
            raise ValueError("max_updates cannot exceed 64")
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._max_updates = max_updates
        self._max_delay_seconds = max_delay_ms / 1000
        self._clock = clock
        self._pending: dict[WeightKey, _PendingState] = {}
        self._changed = asyncio.Event()
        self._flush_complete = asyncio.Event()
        self._flush_complete.set()
        self._flush_requested = False
        self._change_index = 0
        self._last_report_sequence = 0
        self._last_acknowledged_sequence = 0

    @property
    def last_report_sequence(self) -> int:
        """Return the latest sequence allocated for this Worker process start."""

        return self._last_report_sequence

    def mark_changes(self) -> int:
        """Mark the pending boundary before acquiring a full Manager snapshot."""

        return self._change_index

    def record(self, change: ExpertStateChange) -> None:
        """Coalesce one state change without blocking the Weight Manager."""

        if not isinstance(change, ExpertStateChange):
            raise TypeError("state reporter requires ExpertStateChange values")
        _validate_positive_uint64(
            "placement_generation",
            change.placement_generation,
        )
        self._validate_key(change.expert.key)
        self._change_index += 1
        previous = self._pending.get(change.expert.key)
        self._pending[change.expert.key] = _PendingState(
            change_index=self._change_index,
            recorded_at=self._clock() if previous is None else previous.recorded_at,
            change=change,
        )
        self._flush_complete.clear()
        self._changed.set()

    async def flush(self) -> int:
        """Make pending changes immediately available and wait until they are taken.

        Returns:
            The last report sequence allocated after the pending set becomes empty.
            The caller must separately ensure that messages through this sequence
            have entered its outgoing stream before sending dependent control data.
        """

        while self._pending:
            self._flush_requested = True
            self._changed.set()
            self._flush_complete.clear()
            if not self._pending:
                break
            await self._flush_complete.wait()
        return self._last_report_sequence

    def full_state_parts(
        self,
        placement_generation: int,
        states: Iterable[ExpertState],
        *,
        discard_pending_through: int,
    ) -> tuple[weight_control_pb2.WorkerWeightMessage, ...]:
        """Encode one complete snapshot while retaining later state changes."""

        _validate_positive_uint64("placement_generation", placement_generation)
        if not 0 <= discard_pending_through <= self._change_index:
            raise ValueError("snapshot pending boundary is invalid")
        resolved = tuple(states)
        seen: set[WeightKey] = set()
        for state in resolved:
            if not isinstance(state, ExpertState):
                raise TypeError("full state must contain ExpertState values")
            self._validate_key(state.key)
            if state.key in seen:
                raise ValueError("full state contains a duplicate expert")
            seen.add(state.key)
        for key, pending in tuple(self._pending.items()):
            if pending.change_index <= discard_pending_through:
                self._pending.pop(key)
        self._refresh_changed_event()

        sequence = self._next_sequence()
        part_count = max(
            1,
            (len(resolved) + MAX_EXPERTS_PER_CONTROL_PART - 1) // MAX_EXPERTS_PER_CONTROL_PART,
        )
        messages: list[weight_control_pb2.WorkerWeightMessage] = []
        for part_index in range(part_count):
            start = part_index * MAX_EXPERTS_PER_CONTROL_PART
            part_states = resolved[start : start + MAX_EXPERTS_PER_CONTROL_PART]
            messages.append(
                weight_control_pb2.WorkerWeightMessage(
                    full_state=weight_control_pb2.FullExpertStatePart(
                        placement_generation=placement_generation,
                        report_sequence=sequence,
                        part_index=part_index,
                        part_count=part_count,
                        experts=[_encode_state(state) for state in part_states],
                    )
                )
            )
        return tuple(messages)

    async def take_updates(
        self,
        *,
        force: bool = False,
    ) -> weight_control_pb2.WorkerWeightMessage | None:
        """Return the next same-generation batch after size or delay threshold."""

        while True:
            selected = self._select_pending()
            if selected:
                oldest = selected[0].recorded_at
                ready = force or self._flush_requested or len(selected) >= self._max_updates
                ready = ready or self._clock() - oldest >= self._max_delay_seconds
                if ready:
                    return self._take_selected(selected)
                remaining = self._max_delay_seconds - (self._clock() - oldest)
            elif force:
                return None
            else:
                remaining = None

            self._changed.clear()
            if remaining is None:
                await self._changed.wait()
            else:
                try:
                    async with asyncio.timeout(max(0.0, remaining)):
                        await self._changed.wait()
                except TimeoutError:
                    pass

    def acknowledge(self, report_sequence: int) -> bool:
        """Advance the Controller acknowledgement monotonically."""

        if isinstance(report_sequence, bool) or not isinstance(report_sequence, int):
            raise ValueError("report_sequence must be an integer")
        if report_sequence <= 0 or report_sequence > self._last_report_sequence:
            raise ValueError("acknowledgement references an unsent report sequence")
        if report_sequence < self._last_acknowledged_sequence:
            raise ValueError("acknowledgement sequence moved backwards")
        if report_sequence == self._last_acknowledged_sequence:
            return False
        self._last_acknowledged_sequence = report_sequence
        return True

    def _select_pending(self) -> tuple[_PendingState, ...]:
        if not self._pending:
            return ()
        ordered = sorted(self._pending.values(), key=lambda pending: pending.change_index)
        generation = ordered[0].change.placement_generation
        return tuple(
            pending for pending in ordered if pending.change.placement_generation == generation
        )[: self._max_updates]

    def _take_selected(
        self,
        selected: tuple[_PendingState, ...],
    ) -> weight_control_pb2.WorkerWeightMessage:
        for pending in selected:
            current = self._pending.get(pending.change.expert.key)
            if current is pending:
                self._pending.pop(pending.change.expert.key)
        self._refresh_changed_event()
        sequence = self._next_sequence()
        generation = selected[0].change.placement_generation
        return weight_control_pb2.WorkerWeightMessage(
            state_updates=weight_control_pb2.ExpertStateUpdates(
                placement_generation=generation,
                report_sequence=sequence,
                experts=[_encode_state(pending.change.expert) for pending in selected],
            )
        )

    def _next_sequence(self) -> int:
        self._last_report_sequence += 1
        return self._last_report_sequence

    def _refresh_changed_event(self) -> None:
        if self._pending:
            self._changed.set()
        else:
            self._flush_requested = False
            self._flush_complete.set()
            self._changed.clear()

    def _validate_key(self, key: WeightKey) -> None:
        if key.layer_id >= self._num_layers or key.expert_id >= self._experts_per_layer:
            raise ValueError("expert state exceeds the configured model shape")
