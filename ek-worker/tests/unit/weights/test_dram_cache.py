"""Tests for byte-bounded DRAM cache admission, LRU, and references."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest

from expertkit_worker.weights.dram_cache import DramCache, WeightKey


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run one isolated async cache scenario."""

    return asyncio.run(coroutine)


async def insert(cache: DramCache[object], key: WeightKey, byte_count: int) -> object:
    """Insert one object and return it after releasing the commit lease."""

    value = object()
    reservation = await cache.reserve(byte_count)
    lease = await reservation.commit(key, value, actual_bytes=byte_count)
    await lease.close()
    return value


def test_cache_hit_retains_value_and_updates_lru_order() -> None:
    async def scenario() -> None:
        cache: DramCache[object] = DramCache(20)
        first = await insert(cache, WeightKey(0, 0), 10)
        await insert(cache, WeightKey(0, 1), 10)

        lease = await cache.acquire(WeightKey(0, 0))
        assert lease is not None
        assert lease.value is first
        await lease.close()
        reservation = await cache.reserve(10)
        third = object()
        third_lease = await reservation.commit(WeightKey(0, 2), third, actual_bytes=10)
        await third_lease.close()

        assert await cache.acquire(WeightKey(0, 1)) is None
        first_lease = await cache.acquire(WeightKey(0, 0))
        assert first_lease is not None
        await first_lease.close()

    run(scenario())


def test_referenced_entry_makes_reservation_wait_until_release() -> None:
    async def scenario() -> None:
        cache: DramCache[object] = DramCache(10)
        await insert(cache, WeightKey(0, 0), 10)
        lease = await cache.acquire(WeightKey(0, 0))
        assert lease is not None

        waiting = asyncio.create_task(cache.reserve(10))
        await asyncio.sleep(0)
        assert waiting.done() is False
        await lease.close()
        reservation = await asyncio.wait_for(waiting, timeout=1)
        await reservation.cancel()

    run(scenario())


def test_reservation_uses_actual_bytes_and_cancel_returns_budget() -> None:
    async def scenario() -> None:
        cache: DramCache[object] = DramCache(20)
        reservation = await cache.reserve(20)
        lease = await reservation.commit(WeightKey(1, 2), object(), actual_bytes=7)
        await lease.close()
        stats = await cache.stats()
        assert stats.used_bytes == 7
        assert stats.reserved_bytes == 0

        cancelled = await cache.reserve(13)
        assert (await cache.stats()).reserved_bytes == 13
        await cancelled.cancel()
        assert (await cache.stats()).reserved_bytes == 0

    run(scenario())


def test_commit_rejects_entry_larger_than_reservation_without_leaking_budget() -> None:
    async def scenario() -> None:
        cache: DramCache[object] = DramCache(20)
        reservation = await cache.reserve(8)

        with pytest.raises(ValueError, match="exceeds its reservation"):
            await reservation.commit(WeightKey(0, 0), object(), actual_bytes=9)

        assert (await cache.stats()).reserved_bytes == 0

    run(scenario())


def test_remove_and_clear_refuse_active_references_or_reservations() -> None:
    async def scenario() -> None:
        cache: DramCache[object] = DramCache(20)
        await insert(cache, WeightKey(0, 0), 10)
        lease = await cache.acquire(WeightKey(0, 0))
        assert lease is not None
        assert await cache.remove(WeightKey(0, 0)) is False
        with pytest.raises(RuntimeError, match="active references"):
            await cache.clear()
        await lease.close()
        assert await cache.remove(WeightKey(0, 0)) is True

        reservation = await cache.reserve(10)
        with pytest.raises(RuntimeError, match="active reservations"):
            await cache.clear()
        await reservation.cancel()
        await cache.clear()

    run(scenario())


def test_one_reservation_cannot_exceed_cache_capacity() -> None:
    async def scenario() -> None:
        cache: DramCache[object] = DramCache(10)
        with pytest.raises(ValueError, match="exceeds max_bytes"):
            await cache.reserve(11)

    run(scenario())
