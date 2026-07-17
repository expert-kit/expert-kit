"""Tests for direct-I/O disk-cache identity and lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import expertkit_worker.weights.disk_cache as disk_cache_module
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


def test_direct_io_disk_cache_initializes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized: list[Path] = []
    monkeypatch.setattr(
        disk_cache_module,
        "initialize_direct_io_directory",
        initialized.append,
    )
    cache = DirectIOWeightDiskCache(
        root=tmp_path,
        model_name="fixture/model",
        max_concurrent_operations=1,
    )

    async def scenario() -> None:
        await asyncio.gather(cache.initialize(), cache.initialize())
        await cache.initialize()

    try:
        asyncio.run(scenario())
        assert initialized == [tmp_path / "fixture" / "model"]
    finally:
        cache.close()


def test_direct_io_disk_cache_surfaces_initialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_path: Path) -> None:
        raise OSError("direct I/O unavailable")

    monkeypatch.setattr(disk_cache_module, "initialize_direct_io_directory", fail)
    cache = DirectIOWeightDiskCache(
        root=tmp_path,
        model_name="Qwen",
        max_concurrent_operations=1,
    )
    try:
        with pytest.raises(OSError, match="direct I/O unavailable"):
            asyncio.run(cache.initialize())
    finally:
        cache.close()
