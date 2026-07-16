"""Tests for direct-I/O disk-cache identity and lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from expertkit_worker.weights.disk_cache import DirectIOWeightDiskCache
from expertkit_worker.weights.dram_cache import WeightKey


def test_direct_io_disk_cache_uses_existing_expert_path(tmp_path: Path) -> None:
    cache = DirectIOWeightDiskCache(
        root=tmp_path,
        model_name="Qwen",
        max_concurrent_operations=2,
    )
    try:
        assert cache.path(WeightKey(3, 9)) == tmp_path / "Qwen" / "l3-e9"
    finally:
        cache.close()
        cache.close()


def test_direct_io_disk_cache_rejects_operations_after_close(tmp_path: Path) -> None:
    cache = DirectIOWeightDiskCache(
        root=tmp_path,
        model_name="Qwen",
        max_concurrent_operations=1,
    )
    cache.close()

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(cache.read(WeightKey(0, 0), max_bytes=128))
