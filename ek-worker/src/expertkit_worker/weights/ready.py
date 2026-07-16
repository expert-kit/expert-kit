"""Direct ready-weight lookup with all-or-nothing in-flight retention."""

from __future__ import annotations

import threading
from dataclasses import dataclass


class WeightsNotReady(RuntimeError):
    """Report required experts that cannot accept a new computation."""

    def __init__(self, expert_ids: tuple[int, ...]) -> None:
        super().__init__("required expert weights are not ready")
        self.expert_ids = expert_ids


@dataclass(slots=True)
class _ReadyEntry[T]:
    value: T
    accepting: bool = True
    use_count: int = 0


class ReadyWeightLease[T]:
    """Hold direct Backend objects and their usage counts until explicit release."""

    def __init__(
        self,
        table: ReadyWeightTable[T],
        layer_id: int,
        expert_ids: tuple[int, ...],
        entries: tuple[_ReadyEntry[T], ...],
    ) -> None:
        self._table = table
        self._layer_id = layer_id
        self.expert_ids = expert_ids
        self.objects = tuple(entry.value for entry in entries)
        self._entries = entries
        self._close_lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        """Decrement every retained usage count exactly once."""

        with self._close_lock:
            if self._closed:
                return
            self._table._release(self._layer_id, self.expert_ids, self._entries)
            self._closed = True


class ReadyWeightTable[T]:
    """Store the only computation-ready object for every model expert position."""

    def __init__(self, num_layers: int, experts_per_layer: int) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("experts_per_layer", experts_per_layer),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._num_layers = num_layers
        self._experts_per_layer = experts_per_layer
        self._entries: list[list[_ReadyEntry[T] | None]] = [
            [None] * experts_per_layer for _ in range(num_layers)
        ]
        self._condition = threading.Condition(threading.Lock())

    def publish(self, layer_id: int, expert_id: int, value: T) -> None:
        """Make one fully prepared Backend object available to new computation."""

        self._validate_position(layer_id, expert_id)
        if value is None:
            raise ValueError("ready weight value must not be None")
        with self._condition:
            if self._entries[layer_id][expert_id] is not None:
                raise RuntimeError("ready weight position is already occupied")
            self._entries[layer_id][expert_id] = _ReadyEntry(value)
            self._condition.notify_all()

    def acquire_many(
        self,
        layer_id: int,
        distinct_expert_ids: tuple[int, ...],
    ) -> ReadyWeightLease[T]:
        """Retain all requested ready objects without loading, copying, or waiting.

        Args:
            layer_id: Layer that scopes the stable expert numbers.
            distinct_expert_ids: Sorted unique valid expert numbers already derived
                from the received routing Tensor.

        Returns:
            Direct Backend-native objects aligned with ``distinct_expert_ids``.

        Raises:
            WeightsNotReady: At least one object is missing or withdrawing. No usage
                count changes when this happens.
            ValueError: Layer or expert metadata violates the configured table shape.
        """

        expert_ids = tuple(distinct_expert_ids)
        self._validate_expert_list(layer_id, expert_ids)
        with self._condition:
            entries = tuple(self._entries[layer_id][expert_id] for expert_id in expert_ids)
            unavailable = tuple(
                expert_id
                for expert_id, entry in zip(expert_ids, entries, strict=True)
                if entry is None or not entry.accepting
            )
            if unavailable:
                raise WeightsNotReady(unavailable)
            retained = tuple(entry for entry in entries if entry is not None)
            for entry in retained:
                entry.use_count += 1
            return ReadyWeightLease(self, layer_id, expert_ids, retained)

    def begin_withdrawal(self, layer_id: int, expert_id: int) -> None:
        """Prevent new acquisition while existing leases retain the object."""

        self._validate_position(layer_id, expert_id)
        with self._condition:
            entry = self._entries[layer_id][expert_id]
            if entry is None:
                raise WeightsNotReady((expert_id,))
            entry.accepting = False
            self._condition.notify_all()

    def finish_withdrawal(
        self,
        layer_id: int,
        expert_id: int,
        *,
        timeout: float | None = None,
    ) -> T:
        """Wait for active leases, remove the object, and return it to cache policy.

        Warning:
            This is a blocking lifecycle operation. Async Weight Manager code must
            call it in its bounded loading or control executor, never on an asyncio
            event-loop thread.
        """

        self._validate_position(layer_id, expert_id)
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        with self._condition:
            entry = self._entries[layer_id][expert_id]
            if entry is None:
                raise WeightsNotReady((expert_id,))
            if entry.accepting:
                raise RuntimeError("begin_withdrawal must run before finish_withdrawal")
            if not self._condition.wait_for(lambda: entry.use_count == 0, timeout=timeout):
                raise TimeoutError("timed out waiting for ready weight users")
            if self._entries[layer_id][expert_id] is not entry:
                raise RuntimeError("ready weight position changed during withdrawal")
            self._entries[layer_id][expert_id] = None
            self._condition.notify_all()
            return entry.value

    def usage_count(self, layer_id: int, expert_id: int) -> int:
        """Return the current in-flight count for lifecycle tests and reporting."""

        self._validate_position(layer_id, expert_id)
        with self._condition:
            entry = self._entries[layer_id][expert_id]
            return 0 if entry is None else entry.use_count

    def is_ready(self, layer_id: int, expert_id: int) -> bool:
        """Return whether one position currently accepts new computation."""

        self._validate_position(layer_id, expert_id)
        with self._condition:
            entry = self._entries[layer_id][expert_id]
            return entry is not None and entry.accepting

    def _release(
        self,
        layer_id: int,
        expert_ids: tuple[int, ...],
        entries: tuple[_ReadyEntry[T], ...],
    ) -> None:
        with self._condition:
            for expert_id, entry in zip(expert_ids, entries, strict=True):
                if self._entries[layer_id][expert_id] is not entry:
                    raise RuntimeError("retained ready weight changed before release")
                if entry.use_count <= 0:
                    raise RuntimeError("ready weight usage count underflow")
            for entry in entries:
                entry.use_count -= 1
            self._condition.notify_all()

    def _validate_expert_list(self, layer_id: int, expert_ids: tuple[int, ...]) -> None:
        self._validate_layer(layer_id)
        previous = -1
        for expert_id in expert_ids:
            self._validate_expert(expert_id)
            if expert_id <= previous:
                raise ValueError("distinct_expert_ids must be sorted with no duplicates")
            previous = expert_id

    def _validate_position(self, layer_id: int, expert_id: int) -> None:
        self._validate_layer(layer_id)
        self._validate_expert(expert_id)

    def _validate_layer(self, layer_id: int) -> None:
        if (
            isinstance(layer_id, bool)
            or not isinstance(layer_id, int)
            or not 0 <= layer_id < self._num_layers
        ):
            raise ValueError("layer_id exceeds the ready weight table")

    def _validate_expert(self, expert_id: int) -> None:
        if (
            isinstance(expert_id, bool)
            or not isinstance(expert_id, int)
            or not 0 <= expert_id < self._experts_per_layer
        ):
            raise ValueError("expert_id exceeds the ready weight table")
