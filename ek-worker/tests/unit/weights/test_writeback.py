"""Tests for bounded asynchronous expert disk writeback."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from expertkit_worker.weights.direct_io import AlignedWeightBuffer
from expertkit_worker.weights.disk_cache import WeightDiskCache
from expertkit_worker.weights.dram_cache import DramCache, WeightKey
from expertkit_worker.weights.loader import CachedCpuWeight, CpuWeightLease, WeightSource
from expertkit_worker.weights.writeback import DiskWriteback


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run one isolated writeback scenario."""

    return asyncio.run(coroutine)


def make_buffer(payload: bytes) -> AlignedWeightBuffer:
    """Copy one test payload into aligned application-owned memory."""

    result = AlignedWeightBuffer(len(payload))
    view = result.view()
    try:
        view[:] = payload
    finally:
        view.release()
    return result


async def make_lease(
    cache: DramCache[CachedCpuWeight[object]],
    key: WeightKey,
    payload: bytes,
) -> CpuWeightLease[object]:
    """Insert and retain one minimal cache entry for writeback."""

    buffer = make_buffer(payload)
    entry = CachedCpuWeight(
        buffer=buffer,
        parsed=None,  # type: ignore[arg-type]
        value=object(),
        byte_count=len(payload),
    )
    reservation = await cache.reserve(len(payload))
    cache_lease = await reservation.commit(key, entry, actual_bytes=len(payload))
    return CpuWeightLease(cache_lease, WeightSource.WEIGHT_SERVER)


class FakeDiskCache(WeightDiskCache):
    """Capture writes and optionally hold or fail the writer."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.allow = asyncio.Event()
        self.allow.set()
        self.failure: Exception | None = None
        self.writes: list[tuple[WeightKey, bytes]] = []

    async def read(self, key: WeightKey, *, max_bytes: int) -> AlignedWeightBuffer:
        raise NotImplementedError

    async def remove(self, key: WeightKey) -> None:
        raise NotImplementedError

    async def write(self, key: WeightKey, buffer: AlignedWeightBuffer) -> None:
        self.started.set()
        await self.allow.wait()
        if self.failure is not None:
            raise self.failure
        view = buffer.view()
        try:
            self.writes.append((key, bytes(view)))
        finally:
            view.release()


def test_writeback_publishes_buffer_and_releases_cache_lease() -> None:
    async def scenario() -> None:
        key = WeightKey(1, 2)
        cache: DramCache[CachedCpuWeight[object]] = DramCache(64)
        lease = await make_lease(cache, key, b"weight")
        disk = FakeDiskCache()
        writeback: DiskWriteback[object] = DiskWriteback(
            disk_cache=disk,
            enabled=True,
            max_pending=2,
        )
        writeback.start()

        assert writeback.try_submit(key, lease) is True
        await writeback.close()

        assert disk.writes == [(key, b"weight")]
        assert await cache.remove(key) is True

    run(scenario())


def test_writeback_failure_does_not_retain_cpu_weight() -> None:
    async def scenario() -> None:
        key = WeightKey(0, 0)
        cache: DramCache[CachedCpuWeight[object]] = DramCache(64)
        lease = await make_lease(cache, key, b"weight")
        disk = FakeDiskCache()
        disk.failure = OSError("disk full")
        writeback: DiskWriteback[object] = DiskWriteback(
            disk_cache=disk,
            enabled=True,
            max_pending=1,
        )
        writeback.start()

        assert writeback.try_submit(key, lease) is True
        await writeback.close()

        assert await cache.remove(key) is True

    run(scenario())


def test_writeback_rejects_new_job_when_bounded_queue_is_full() -> None:
    async def scenario() -> None:
        keys = (WeightKey(0, 0), WeightKey(0, 1), WeightKey(0, 2))
        cache: DramCache[CachedCpuWeight[object]] = DramCache(64)
        leases = [await make_lease(cache, key, b"weight") for key in keys]
        disk = FakeDiskCache()
        disk.allow.clear()
        writeback: DiskWriteback[object] = DiskWriteback(
            disk_cache=disk,
            enabled=True,
            max_pending=1,
        )
        writeback.start()

        assert writeback.try_submit(keys[0], leases[0]) is True
        await asyncio.wait_for(disk.started.wait(), timeout=1)
        assert writeback.try_submit(keys[1], leases[1]) is True
        assert writeback.try_submit(keys[2], leases[2]) is False
        await leases[2].close()
        disk.allow.set()
        await writeback.close()

        for key in keys:
            assert await cache.remove(key) is True

    run(scenario())


def test_disabled_writeback_leaves_lease_with_caller() -> None:
    async def scenario() -> None:
        key = WeightKey(0, 0)
        cache: DramCache[CachedCpuWeight[object]] = DramCache(64)
        lease = await make_lease(cache, key, b"weight")
        writeback: DiskWriteback[object] = DiskWriteback(
            disk_cache=FakeDiskCache(),
            enabled=False,
            max_pending=1,
        )
        writeback.start()

        assert writeback.try_submit(key, lease) is False
        assert await cache.remove(key) is False
        await lease.close()
        await writeback.close()
        assert await cache.remove(key) is True

    run(scenario())
