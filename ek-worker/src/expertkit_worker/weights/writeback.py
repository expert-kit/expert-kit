"""Bounded asynchronous writeback of validated remote expert files."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from expertkit_worker.weights.disk_cache import WeightDiskCache
from expertkit_worker.weights.dram_cache import WeightKey
from expertkit_worker.weights.loader import CpuWeightLease

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _WriteJob[CpuWeightT]:
    key: WeightKey
    lease: CpuWeightLease[CpuWeightT]


class DiskWriteback[CpuWeightT]:
    """Write remote weights without delaying their transition to ready."""

    def __init__(
        self,
        *,
        disk_cache: WeightDiskCache,
        enabled: bool,
        max_pending: int,
    ) -> None:
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        self._disk_cache = disk_cache
        self._enabled = enabled
        self._queue: asyncio.Queue[_WriteJob[CpuWeightT] | None] = asyncio.Queue(
            maxsize=max_pending
        )
        self._task: asyncio.Task[None] | None = None
        self._accepting = False

    def start(self) -> None:
        """Start the single bounded writer in the current event loop."""

        if not self._enabled or self._task is not None:
            return
        self._accepting = True
        self._task = asyncio.create_task(self._run(), name="weight-disk-writeback")

    def try_submit(self, key: WeightKey, lease: CpuWeightLease[CpuWeightT]) -> bool:
        """Transfer lease ownership when the bounded write queue has capacity.

        Returns:
            ``True`` when the writer now owns and will release ``lease``. The
            caller retains ownership when this method returns ``False``.
        """

        if not self._enabled or not self._accepting:
            return False
        try:
            self._queue.put_nowait(_WriteJob(key, lease))
        except asyncio.QueueFull:
            logger.warning(
                "weight_disk_writeback_full",
                layer_id=key.layer_id,
                expert_id=key.expert_id,
            )
            return False
        return True

    async def close(self) -> None:
        """Stop admission, drain accepted writes, and release every lease."""

        task = self._task
        if task is None:
            return
        self._accepting = False
        await self._queue.join()
        await self._queue.put(None)
        await task
        self._task = None

    async def _run(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job is None:
                    return
                try:
                    await self._disk_cache.write(job.key, job.lease.cached.buffer)
                except Exception:
                    logger.warning(
                        "weight_disk_writeback_failed",
                        layer_id=job.key.layer_id,
                        expert_id=job.key.expert_id,
                        exc_info=True,
                    )
                finally:
                    await job.lease.close()
            finally:
                self._queue.task_done()
