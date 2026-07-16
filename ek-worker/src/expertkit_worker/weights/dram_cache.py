"""Byte-bounded least-recently-used cache for parsed CPU expert weights."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class WeightKey:
    """Identify one expert within the Worker's fixed model instance."""

    layer_id: int
    expert_id: int

    def __post_init__(self) -> None:
        for name, value in (("layer_id", self.layer_id), ("expert_id", self.expert_id)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(slots=True)
class _CacheRecord[T]:
    value: T
    byte_count: int
    references: int = 0


@dataclass(frozen=True, slots=True)
class DramCacheStats:
    """Report current logical Host-memory accounting."""

    max_bytes: int
    used_bytes: int
    reserved_bytes: int
    entry_count: int


class DramCacheLease[T]:
    """Keep one cached CPU weight in memory during an external operation."""

    def __init__(self, cache: DramCache[T], key: WeightKey, value: T) -> None:
        self._cache = cache
        self._key = key
        self.value = value
        self._closed = False

    async def close(self) -> None:
        """Release this cache reference exactly once."""

        if self._closed:
            return
        self._closed = True
        await self._cache._release(self._key)

    async def __aenter__(self) -> DramCacheLease[T]:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()


class DramReservation[T]:
    """Hold Host-memory budget before a read or download allocates its buffer."""

    def __init__(self, cache: DramCache[T], reserved_bytes: int) -> None:
        self._cache = cache
        self.reserved_bytes = reserved_bytes
        self._finished = False

    async def commit(
        self,
        key: WeightKey,
        value: T,
        *,
        actual_bytes: int,
    ) -> DramCacheLease[T]:
        """Install the completed cache entry and retain it for the caller."""

        if self._finished:
            raise RuntimeError("DRAM cache reservation is already finished")
        self._finished = True
        return await self._cache._commit(self, key, value, actual_bytes)

    async def cancel(self) -> None:
        """Return unused budget after a failed or superseded operation."""

        if self._finished:
            return
        self._finished = True
        await self._cache._cancel(self)

    async def __aenter__(self) -> DramReservation[T]:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.cancel()


class DramCache[T]:
    """Keep parsed CPU weights within one explicit logical byte limit."""

    def __init__(self, max_bytes: int) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._max_bytes = max_bytes
        self._used_bytes = 0
        self._reserved_bytes = 0
        self._entries: OrderedDict[WeightKey, _CacheRecord[T]] = OrderedDict()
        self._condition = asyncio.Condition()

    async def acquire(self, key: WeightKey) -> DramCacheLease[T] | None:
        """Retain one entry and update its LRU position, or report a cache miss."""

        async with self._condition:
            record = self._entries.get(key)
            if record is None:
                return None
            record.references += 1
            self._entries.move_to_end(key)
            return DramCacheLease(self, key, record.value)

    async def reserve(self, byte_count: int) -> DramReservation[T]:
        """Wait for Host budget after evicting eligible least-recently-used entries."""

        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("byte_count must be a positive integer")
        if byte_count > self._max_bytes:
            raise ValueError("one DRAM cache reservation exceeds max_bytes")

        async with self._condition:
            while True:
                self._evict_until_available(byte_count)
                if self._available_bytes >= byte_count:
                    self._reserved_bytes += byte_count
                    return DramReservation(self, byte_count)
                await self._condition.wait()

    async def remove(self, key: WeightKey) -> bool:
        """Remove one unreferenced entry without waiting for active users."""

        async with self._condition:
            record = self._entries.get(key)
            if record is None:
                return False
            if record.references:
                return False
            self._entries.pop(key)
            self._used_bytes -= record.byte_count
            self._condition.notify_all()
            return True

    async def stats(self) -> DramCacheStats:
        """Return a consistent accounting snapshot."""

        async with self._condition:
            return DramCacheStats(
                max_bytes=self._max_bytes,
                used_bytes=self._used_bytes,
                reserved_bytes=self._reserved_bytes,
                entry_count=len(self._entries),
            )

    async def clear(self) -> None:
        """Remove all entries after peer serving and loading have stopped."""

        async with self._condition:
            if self._reserved_bytes:
                raise RuntimeError("cannot clear DRAM cache with active reservations")
            if any(record.references for record in self._entries.values()):
                raise RuntimeError("cannot clear DRAM cache with active references")
            self._entries.clear()
            self._used_bytes = 0
            self._condition.notify_all()

    @property
    def _available_bytes(self) -> int:
        return self._max_bytes - self._used_bytes - self._reserved_bytes

    def _evict_until_available(self, required_bytes: int) -> None:
        while self._available_bytes < required_bytes:
            victim = next(
                (key for key, record in self._entries.items() if record.references == 0),
                None,
            )
            if victim is None:
                return
            record = self._entries.pop(victim)
            self._used_bytes -= record.byte_count

    async def _release(self, key: WeightKey) -> None:
        async with self._condition:
            record = self._entries.get(key)
            if record is None or record.references <= 0:
                raise RuntimeError("DRAM cache reference is no longer valid")
            record.references -= 1
            self._condition.notify_all()

    async def _commit(
        self,
        reservation: DramReservation[T],
        key: WeightKey,
        value: T,
        actual_bytes: int,
    ) -> DramCacheLease[T]:
        if isinstance(actual_bytes, bool) or not isinstance(actual_bytes, int) or actual_bytes <= 0:
            await self._cancel(reservation)
            raise ValueError("actual_bytes must be a positive integer")
        async with self._condition:
            if reservation.reserved_bytes > self._reserved_bytes:
                raise RuntimeError("DRAM cache reservation accounting underflow")
            self._reserved_bytes -= reservation.reserved_bytes
            if actual_bytes > reservation.reserved_bytes:
                self._condition.notify_all()
                raise ValueError("actual cache entry exceeds its reservation")
            if key in self._entries:
                self._condition.notify_all()
                raise RuntimeError("DRAM cache key is already present")
            record = _CacheRecord(value, actual_bytes, references=1)
            self._entries[key] = record
            self._used_bytes += actual_bytes
            self._condition.notify_all()
            return DramCacheLease(self, key, value)

    async def _cancel(self, reservation: DramReservation[T]) -> None:
        async with self._condition:
            if reservation.reserved_bytes > self._reserved_bytes:
                raise RuntimeError("DRAM cache reservation accounting underflow")
            self._reserved_bytes -= reservation.reserved_bytes
            self._condition.notify_all()
