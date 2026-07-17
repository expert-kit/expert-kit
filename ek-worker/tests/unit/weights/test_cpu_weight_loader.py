"""Tests for ordered CPU weight lookup, validation, and caching."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest
import torch
from safetensors.torch import save as official_save

from expertkit_worker.backends.torch import TorchWeightAdapter
from expertkit_worker.weights.direct_io import AlignedWeightBuffer, InvalidWeightFile
from expertkit_worker.weights.disk_cache import WeightDiskCache
from expertkit_worker.weights.dram_cache import DramCache, WeightKey
from expertkit_worker.weights.loader import (
    CachedCpuWeight,
    CpuWeightLoader,
    WeightLoadFailed,
    WeightSource,
)
from expertkit_worker.weights.transfer import (
    WeightNotFound,
    WeightTransfer,
    WeightTransferError,
)

_MODEL_NAME = "Qwen/Test Model"
_WEIGHT_SERVER = "http://weights.internal:8080"
_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 3


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run one isolated async loading scenario."""

    return asyncio.run(coroutine)


def make_payload(*, gate_value: float = 1.0) -> bytes:
    """Return one valid standard per-expert SafeTensors file."""

    return official_save(
        {
            "model.gate_proj.weight": torch.full(
                (_INTERMEDIATE_DIM, _HIDDEN_DIM), gate_value, dtype=torch.float32
            ),
            "model.up_proj.weight": torch.full(
                (_INTERMEDIATE_DIM, _HIDDEN_DIM), 2, dtype=torch.float32
            ),
            "model.down_proj.weight": torch.full(
                (_HIDDEN_DIM, _INTERMEDIATE_DIM), 3, dtype=torch.float32
            ),
        }
    )


def make_buffer(payload: bytes) -> AlignedWeightBuffer:
    """Copy test bytes into one application-owned aligned buffer."""

    result = AlignedWeightBuffer(len(payload))
    view = result.view()
    try:
        view[:] = payload
    finally:
        view.release()
    return result


class FakeTransfer(WeightTransfer):
    """Return configured payloads or failures while recording source order."""

    def __init__(self, results: dict[str, bytes | WeightTransferError]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def start(self) -> None:
        pass

    async def download(self, url: str, *, max_bytes: int) -> AlignedWeightBuffer:
        self.calls.append(url)
        result = self.results[url]
        if isinstance(result, WeightTransferError):
            raise result
        if len(result) > max_bytes:
            raise AssertionError("fixture exceeds loader byte bound")
        return make_buffer(result)

    async def close(self) -> None:
        pass


class BlockingTransfer(FakeTransfer):
    """Hold one transfer so concurrent callers can join the same load."""

    def __init__(self, results: dict[str, bytes | WeightTransferError]) -> None:
        super().__init__(results)
        self.started = asyncio.Event()
        self.allow = asyncio.Event()

    async def download(self, url: str, *, max_bytes: int) -> AlignedWeightBuffer:
        self.calls.append(url)
        self.started.set()
        await self.allow.wait()
        result = self.results[url]
        if isinstance(result, WeightTransferError):
            raise result
        if len(result) > max_bytes:
            raise AssertionError("fixture exceeds loader byte bound")
        return make_buffer(result)


class FakeDiskCache(WeightDiskCache):
    """Return configured local files while recording reads and invalidations."""

    def __init__(self, results: dict[WeightKey, bytes | OSError] | None = None) -> None:
        self.results = {} if results is None else results
        self.reads: list[WeightKey] = []
        self.removed: list[WeightKey] = []

    async def read(self, key: WeightKey, *, max_bytes: int) -> AlignedWeightBuffer:
        self.reads.append(key)
        result = self.results.get(key, FileNotFoundError("expert file does not exist"))
        if isinstance(result, OSError):
            raise result
        if len(result) > max_bytes:
            raise AssertionError("disk fixture exceeds loader byte bound")
        return make_buffer(result)

    async def remove(self, key: WeightKey) -> None:
        self.removed.append(key)
        self.results.pop(key, None)

    async def write(self, key: WeightKey, buffer: AlignedWeightBuffer) -> None:
        view = buffer.view()
        try:
            self.results[key] = bytes(view)
        finally:
            view.release()


class BlockingDiskCache(FakeDiskCache):
    """Hold one disk read so concurrent callers overlap before cache insertion."""

    def __init__(self, results: dict[WeightKey, bytes | OSError] | None = None) -> None:
        super().__init__(results)
        self.started = asyncio.Event()
        self.allow = asyncio.Event()

    async def read(self, key: WeightKey, *, max_bytes: int) -> AlignedWeightBuffer:
        self.reads.append(key)
        self.started.set()
        await self.allow.wait()
        result = self.results.get(key, FileNotFoundError("expert file does not exist"))
        if isinstance(result, OSError):
            raise result
        if len(result) > max_bytes:
            raise AssertionError("disk fixture exceeds loader byte bound")
        return make_buffer(result)


def make_loader(
    transfer: WeightTransfer,
    disk_cache: WeightDiskCache | None = None,
    source_results: list[tuple[str, bool]] | None = None,
    *,
    cache_entries: int = 1,
) -> CpuWeightLoader[object, object]:
    """Return a small Torch CPU loader with full parser-headroom capacity."""

    adapter = TorchWeightAdapter(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=torch.float32,
        compute_dtype=torch.float32,
        device="cpu",
    )
    max_bytes = 8 + 16 * 1024 * 1024 + adapter.source_tensor_bytes()
    cache: DramCache[CachedCpuWeight[object]] = DramCache(max_bytes * cache_entries)
    return CpuWeightLoader(
        model_name=_MODEL_NAME,
        disk_cache=FakeDiskCache() if disk_cache is None else disk_cache,
        weight_server_endpoint=_WEIGHT_SERVER,
        adapter=adapter,
        cache=cache,
        transfer=transfer,
        source_result=(
            None
            if source_results is None
            else lambda source, success: source_results.append((source, success))
        ),
    )


def expert_url(endpoint: str, key: WeightKey) -> str:
    """Return the encoded URL expected from the loader fixture."""

    return f"{endpoint}/expert/Qwen%2FTest%20Model/{key.layer_id}/{key.expert_id}"


def test_loader_reuses_parsed_dram_entry_without_repeating_transfer() -> None:
    key = WeightKey(2, 7)
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = FakeTransfer({central_url: make_payload()})
    loader = make_loader(transfer)

    async def scenario() -> None:
        first = await loader.acquire(key)
        assert first.source is WeightSource.WEIGHT_SERVER
        await first.close()

        second = await loader.acquire(
            key,
            peer_endpoints=("http://unused-peer",),
        )
        assert second.source is WeightSource.DRAM
        assert second.cached.value.gate_proj[0, 0].item() == 1
        await second.close()

    run(scenario())
    assert transfer.calls == [central_url]


def test_loader_uses_valid_disk_file_before_remote_sources() -> None:
    key = WeightKey(1, 4)
    disk = FakeDiskCache({key: make_payload(gate_value=5)})
    transfer = FakeTransfer({})
    loader = make_loader(transfer, disk)

    async def scenario() -> None:
        loaded = await loader.acquire(key, peer_endpoints=("http://peer",))
        assert loaded.source is WeightSource.DISK
        assert loaded.cached.value.gate_proj[0, 0].item() == 5
        await loaded.close()

    run(scenario())
    assert disk.reads == [key]
    assert transfer.calls == []


def test_loader_removes_corrupt_disk_file_and_falls_back_to_peer() -> None:
    key = WeightKey(0, 3)
    disk = FakeDiskCache({key: b"not-safetensors"})
    peer = "http://peer.internal:8081"
    peer_url = expert_url(peer, key)
    transfer = FakeTransfer({peer_url: make_payload()})
    loader = make_loader(transfer, disk)

    async def scenario() -> None:
        loaded = await loader.acquire(key, peer_endpoints=(peer,))
        assert loaded.source is WeightSource.PEER
        await loaded.close()

    run(scenario())
    assert disk.removed == [key]
    assert transfer.calls == [peer_url]


def test_loader_tries_each_peer_once_then_the_weight_server() -> None:
    key = WeightKey(3, 9)
    peer_one = "http://peer-1"
    peer_two = "http://peer-2/"
    peer_one_url = expert_url(peer_one, key)
    peer_two_url = expert_url(peer_two.rstrip("/"), key)
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = FakeTransfer(
        {
            peer_one_url: WeightTransferError("peer unavailable", retryable=True),
            peer_two_url: WeightNotFound(),
            central_url: make_payload(gate_value=8),
        }
    )
    source_results: list[tuple[str, bool]] = []
    loader = make_loader(transfer, source_results=source_results)

    async def scenario() -> None:
        loaded = await loader.acquire(
            key,
            peer_endpoints=(peer_one, peer_one, peer_two),
        )
        assert loaded.source is WeightSource.WEIGHT_SERVER
        assert loaded.cached.value.gate_proj[0, 0].item() == 8
        await loaded.close()

    run(scenario())
    assert transfer.calls == [peer_one_url, peer_two_url, central_url]
    assert source_results == [
        ("disk", False),
        ("peer", False),
        ("peer", False),
        ("weight_server", True),
    ]


def test_loader_falls_back_after_invalid_peer_content() -> None:
    key = WeightKey(4, 1)
    peer = "http://peer"
    peer_url = expert_url(peer, key)
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = FakeTransfer(
        {
            peer_url: b"invalid",
            central_url: make_payload(),
        }
    )
    loader = make_loader(transfer)

    async def scenario() -> None:
        loaded = await loader.acquire(key, peer_endpoints=(peer,))
        assert loaded.source is WeightSource.WEIGHT_SERVER
        await loaded.close()

    run(scenario())
    assert transfer.calls == [peer_url, central_url]


def test_loader_removes_disk_file_rejected_before_parsing() -> None:
    key = WeightKey(4, 2)
    disk = FakeDiskCache({key: InvalidWeightFile("file exceeds byte limit")})
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = FakeTransfer({central_url: make_payload()})
    loader = make_loader(transfer, disk)

    async def scenario() -> None:
        loaded = await loader.acquire(key)
        assert loaded.source is WeightSource.WEIGHT_SERVER
        await loaded.close()

    run(scenario())
    assert disk.removed == [key]


def test_loader_reports_all_failures_and_preserves_retryability() -> None:
    key = WeightKey(5, 2)
    peer = "http://peer"
    peer_url = expert_url(peer, key)
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = FakeTransfer(
        {
            peer_url: WeightTransferError("temporary peer failure", retryable=True),
            central_url: WeightNotFound("central weight missing"),
        }
    )
    loader = make_loader(transfer)

    async def scenario() -> None:
        with pytest.raises(WeightLoadFailed) as caught:
            await loader.acquire(key, peer_endpoints=(peer,))
        assert caught.value.retryable is True
        assert [failure.source for failure in caught.value.failures] == [
            WeightSource.DISK,
            WeightSource.PEER,
            WeightSource.WEIGHT_SERVER,
        ]

    run(scenario())


def test_concurrent_same_key_disk_loads_share_one_read_and_return_independent_leases() -> None:
    key = WeightKey(2, 3)
    disk = BlockingDiskCache({key: make_payload()})
    loader = make_loader(FakeTransfer({}), disk, cache_entries=2)

    async def scenario() -> None:
        first = asyncio.create_task(loader.acquire(key))
        second = asyncio.create_task(loader.acquire_local(key))
        await asyncio.wait_for(disk.started.wait(), timeout=1)
        assert disk.reads == [key]
        disk.allow.set()

        first_lease, second_lease = await asyncio.gather(first, second)
        assert first_lease.cached is second_lease.cached
        await first_lease.close()
        assert second_lease.cached.value.gate_proj[0, 0].item() == 1
        await second_lease.close()

    run(scenario())
    assert disk.reads == [key]


def test_cancelling_one_waiter_does_not_cancel_a_shared_remote_load() -> None:
    key = WeightKey(3, 4)
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = BlockingTransfer({central_url: make_payload()})
    loader = make_loader(transfer, cache_entries=2)

    async def scenario() -> None:
        cancelled = asyncio.create_task(loader.acquire(key))
        retained = asyncio.create_task(loader.acquire(key))
        await asyncio.wait_for(transfer.started.wait(), timeout=1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert retained.done() is False

        transfer.allow.set()
        lease = await asyncio.wait_for(retained, timeout=1)
        assert lease.source is WeightSource.WEIGHT_SERVER
        await lease.close()
        assert loader._inflight == {}

    run(scenario())
    assert transfer.calls == [central_url]


def test_concurrent_same_key_failures_are_shared_without_duplicate_source_attempts() -> None:
    key = WeightKey(4, 5)
    central_url = expert_url(_WEIGHT_SERVER, key)
    transfer = BlockingTransfer({central_url: WeightNotFound("missing")})
    loader = make_loader(transfer, cache_entries=2)

    async def scenario() -> None:
        first = asyncio.create_task(loader.acquire(key))
        second = asyncio.create_task(loader.acquire(key))
        await asyncio.wait_for(transfer.started.wait(), timeout=1)
        transfer.allow.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

        assert all(isinstance(result, WeightLoadFailed) for result in results)
        failures = [result for result in results if isinstance(result, WeightLoadFailed)]
        assert failures[0].failures == failures[1].failures
        assert loader._inflight == {}

    run(scenario())
    assert transfer.calls == [central_url]
