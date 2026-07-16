"""Async persistent weight-cache boundary backed by strict direct I/O."""

from __future__ import annotations

import asyncio
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from expertkit_worker.weights.direct_io import (
    AlignedWeightBuffer,
    expert_file_path,
    read_direct,
    write_direct_atomic,
)
from expertkit_worker.weights.dram_cache import WeightKey


class WeightDiskCache(ABC):
    """Read and invalidate local per-expert SafeTensors files asynchronously."""

    @abstractmethod
    async def read(self, key: WeightKey, *, max_bytes: int) -> AlignedWeightBuffer:
        """Read one file into application-owned aligned memory."""

    @abstractmethod
    async def remove(self, key: WeightKey) -> None:
        """Remove one invalid local file if it still exists."""

    @abstractmethod
    async def write(self, key: WeightKey, buffer: AlignedWeightBuffer) -> None:
        """Durably publish one validated SafeTensors buffer."""


class DirectIOWeightDiskCache(WeightDiskCache):
    """Run strict direct-I/O file operations in one bounded thread pool."""

    def __init__(
        self,
        *,
        root: Path,
        model_name: str,
        max_concurrent_operations: int,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if (
            isinstance(max_concurrent_operations, bool)
            or not isinstance(max_concurrent_operations, int)
            or max_concurrent_operations <= 0
        ):
            raise ValueError("max_concurrent_operations must be a positive integer")
        self._root = root
        self._model_name = model_name
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_operations,
            thread_name_prefix="ek-weight-io",
        )
        self._closed = False

    async def read(self, key: WeightKey, *, max_bytes: int) -> AlignedWeightBuffer:
        """Read one complete cache file without using the file page cache."""

        return await self._run_blocking(
            read_direct,
            self.path(key),
            max_bytes=max_bytes,
        )

    async def remove(self, key: WeightKey) -> None:
        """Remove one corrupt cache file without blocking the event loop."""

        await self._run_blocking(self.path(key).unlink, missing_ok=True)

    async def write(self, key: WeightKey, buffer: AlignedWeightBuffer) -> None:
        """Atomically write one validated file through strict direct I/O."""

        await self._run_blocking(write_direct_atomic, self.path(key), buffer)

    def path(self, key: WeightKey) -> Path:
        """Return the deterministic file path for one expert."""

        return expert_file_path(
            self._root,
            self._model_name,
            key.layer_id,
            key.expert_id,
        )

    def close(self) -> None:
        """Stop the blocking-I/O pool after all cache operations have drained."""

        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _run_blocking[ResultT](
        self,
        function: Callable[..., ResultT],
        /,
        *args: object,
        **kwargs: object,
    ) -> ResultT:
        if self._closed:
            raise RuntimeError("weight disk cache is closed")
        loop = asyncio.get_running_loop()
        call = functools.partial(function, *args, **kwargs)
        return await loop.run_in_executor(self._executor, call)
