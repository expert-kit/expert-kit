"""Bounded waiting, drain gates, and idle tracking shared by Worker receivers."""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto

from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import ReceivedWorkerBatch, ReceiverClosed

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class _State(Enum):
    WAITING = auto()
    ACTIVE = auto()


@dataclass(slots=True)
class _Entry:
    item: ReceivedWorkerBatch
    layer_id: int
    expert_ids: tuple[int, ...]
    retained_bytes: int
    state: _State = _State.WAITING


def _validate_experts(
    experts: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    resolved = tuple(experts)
    seen: set[tuple[int, int]] = set()
    for key in resolved:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("each draining expert must contain layer and expert IDs")
        for name, value in zip(("layer_id", "expert_id"), key, strict=True):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _UINT32_MAX
            ):
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        if key in seen:
            raise ValueError("draining experts must not contain duplicates")
        seen.add(key)
    return resolved


class ReceiverQueue:
    """Own the bounded waiting data and drain state for one Worker receiver."""

    def __init__(
        self,
        *,
        max_pending_batches: int,
        max_retained_bytes: int,
        clock: Callable[[], float] = time.monotonic,
        on_pending_changed: Callable[[int], None] | None = None,
    ) -> None:
        for name, value in (
            ("max_pending_batches", max_pending_batches),
            ("max_retained_bytes", max_retained_bytes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name == "max_pending_batches" else 0)
            ):
                qualifier = "positive" if name == "max_pending_batches" else "nonnegative"
                raise ValueError(f"{name} must be a {qualifier} integer")
        self._max_pending_batches = max_pending_batches
        self._max_retained_bytes = max_retained_bytes
        self._clock = clock
        self._on_pending_changed = on_pending_changed or (lambda _count: None)
        self._waiting: deque[_Entry] = deque()
        self._entries: dict[ReceivedWorkerBatch, _Entry] = {}
        self._retained_bytes = 0
        self._active_count = 0
        self._admitted_experts: Counter[tuple[int, int]] = Counter()
        self._draining_experts: dict[tuple[int, int], int] = {}
        self._stop_all_min_topology_version: int | None = None
        self._condition = asyncio.Condition()
        self._closing = False

    @property
    def pending_count(self) -> int:
        """Return the number of admitted batches waiting for execution."""

        return len(self._waiting)

    @property
    def retained_bytes(self) -> int:
        """Return bytes retained by all waiting batch inputs."""

        return self._retained_bytes

    @property
    def active_count(self) -> int:
        """Return batches taken by Worker execution and not yet finished."""

        return self._active_count

    async def admit(
        self,
        item: ReceivedWorkerBatch,
        *,
        retained_bytes: int,
    ) -> TransportError | None:
        """Admit one validated batch or return its drain/capacity rejection."""

        if (
            isinstance(retained_bytes, bool)
            or not isinstance(retained_bytes, int)
            or retained_bytes < 0
        ):
            raise ValueError("retained_bytes must be a nonnegative integer")
        batch = item.batch
        entry = _Entry(
            item=item,
            layer_id=batch.layer_id,
            expert_ids=batch.distinct_expert_ids,
            retained_bytes=retained_bytes,
        )
        async with self._condition:
            if self._closing:
                return TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=True,
                    diagnostic="Worker Transport receiver is closing",
                )
            if item in self._entries:
                raise RuntimeError("received batch is already admitted")
            draining = self._drain_error(entry)
            if draining is not None:
                return draining
            if (
                len(self._waiting) >= self._max_pending_batches
                or self._retained_bytes + retained_bytes > self._max_retained_bytes
            ):
                return TransportError(
                    TransportErrorCode.BUSY,
                    retryable=True,
                    diagnostic="Worker Transport waiting area is full",
                )
            self._waiting.append(entry)
            self._entries[item] = entry
            self._retained_bytes += retained_bytes
            for expert_id in entry.expert_ids:
                self._admitted_experts[(entry.layer_id, expert_id)] += 1
            self._record_pending_count()
            self._condition.notify_all()
        return None

    async def take(self) -> ReceivedWorkerBatch:
        """Move the oldest waiting batch into active execution ownership."""

        async with self._condition:
            await self._condition.wait_for(lambda: self._waiting or self._closing)
            if not self._waiting:
                raise ReceiverClosed("Worker Transport receiver is closed")
            entry = self._waiting.popleft()
            self._retained_bytes -= entry.retained_bytes
            entry.state = _State.ACTIVE
            self._active_count += 1
            self._record_pending_count()
            self._condition.notify_all()
            return entry.item

    async def cancel_waiting(self, item: ReceivedWorkerBatch) -> bool:
        """Remove one cancelled waiting batch; active work remains tracked."""

        async with self._condition:
            entry = self._entries.get(item)
            if entry is None or entry.state is not _State.WAITING:
                return False
            self._waiting.remove(entry)
            self._retained_bytes -= entry.retained_bytes
            self._entries.pop(item)
            self._release_experts(entry)
            self._record_pending_count()
            self._condition.notify_all()
            return True

    async def finish(self, item: ReceivedWorkerBatch) -> None:
        """Release one active batch after response communication is finished."""

        async with self._condition:
            entry = self._entries.get(item)
            if entry is None or entry.state is not _State.ACTIVE:
                raise RuntimeError("Worker batch completion requires an active received batch")
            self._entries.pop(item)
            self._active_count -= 1
            self._release_experts(entry)
            self._condition.notify_all()

    def require_active(self, item: ReceivedWorkerBatch) -> None:
        """Reject completion of a batch that is not currently active."""

        entry = self._entries.get(item)
        if entry is None or entry.state is not _State.ACTIVE:
            raise RuntimeError("Worker batch completion requires an active received batch")

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        """Reject new matching batches while preserving admitted work."""

        selected = _validate_experts(experts)
        if (
            isinstance(min_topology_version, bool)
            or not isinstance(min_topology_version, int)
            or not 0 <= min_topology_version <= _UINT64_MAX
        ):
            raise ValueError("min_topology_version must be a uint64")
        if not isinstance(stop_all, bool):
            raise ValueError("stop_all must be a Boolean")
        if not selected and not stop_all:
            raise ValueError("an expert drain must name at least one expert")

        async with self._condition:
            for key in selected:
                current = self._draining_experts.get(key, 0)
                self._draining_experts[key] = max(current, min_topology_version)
            if stop_all:
                current = self._stop_all_min_topology_version or 0
                self._stop_all_min_topology_version = max(current, min_topology_version)
            self._condition.notify_all()

    async def clear_expert_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        """Clear per-expert gates after later assignments become ready."""

        selected = _validate_experts(experts)
        async with self._condition:
            for key in selected:
                self._draining_experts.pop(key, None)
            self._condition.notify_all()

    async def wait_experts_idle(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        monotonic_deadline: float,
    ) -> None:
        """Wait until no admitted batch names any selected expert."""

        selected = _validate_experts(experts)
        async with self._condition:
            while any(self._admitted_experts[key] for key in selected):
                await self._wait(monotonic_deadline, "admitted expert use")

    async def wait_all_idle(self, *, monotonic_deadline: float) -> None:
        """Wait until no waiting or active batch remains."""

        async with self._condition:
            while self._waiting or self._active_count:
                await self._wait(monotonic_deadline, "all admitted work")

    async def wait_for_pending_count(self, expected: int) -> None:
        """Wait for a pending count used by deterministic tests and diagnostics."""

        async with self._condition:
            await self._condition.wait_for(lambda: len(self._waiting) == expected or self._closing)

    def admitted_count(self, layer_id: int, expert_id: int) -> int:
        """Return waiting plus active batches that name one expert."""

        return self._admitted_experts[(layer_id, expert_id)]

    async def begin_close(self) -> tuple[ReceivedWorkerBatch, ...]:
        """Stop admission and transfer ownership of discarded waiting batches.

        The receiver must finish protocol-specific cancellation for the returned
        batches before calling :meth:`wait_active_empty`. Active batches remain
        valid so Worker execution can finish their response communication.
        """

        async with self._condition:
            if self._closing:
                return ()
            self._closing = True
            waiting = tuple(entry.item for entry in self._waiting)
            for entry in self._waiting:
                self._entries.pop(entry.item)
                self._release_experts(entry)
            self._waiting.clear()
            self._retained_bytes = 0
            self._record_pending_count()
            self._condition.notify_all()
            return waiting

    async def wait_active_empty(self) -> None:
        """Wait until every batch taken before shutdown has finished."""

        async with self._condition:
            await self._condition.wait_for(lambda: self._active_count == 0)

    def _drain_error(self, entry: _Entry) -> TransportError | None:
        min_topology_version = self._stop_all_min_topology_version
        for expert_id in entry.expert_ids:
            expert_version = self._draining_experts.get((entry.layer_id, expert_id))
            if expert_version is not None:
                min_topology_version = max(min_topology_version or 0, expert_version)
        if min_topology_version is None:
            return None
        return TransportError(
            TransportErrorCode.DRAINING,
            retryable=True,
            min_topology_version=min_topology_version,
            diagnostic="Worker is draining the requested expert route",
        )

    def _release_experts(self, entry: _Entry) -> None:
        for expert_id in entry.expert_ids:
            key = (entry.layer_id, expert_id)
            self._admitted_experts[key] -= 1
            if self._admitted_experts[key] == 0:
                del self._admitted_experts[key]

    async def _wait(self, monotonic_deadline: float, subject: str) -> None:
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError(f"deadline expired while waiting for {subject}")
        if math.isinf(remaining):
            await self._condition.wait()
        else:
            async with asyncio.timeout(remaining):
                await self._condition.wait()

    def _record_pending_count(self) -> None:
        with suppress(Exception):
            self._on_pending_changed(len(self._waiting))
