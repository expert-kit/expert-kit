"""Localhost integration tests for the non-recursive peer weight server."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import pytest
import torch
from safetensors.torch import save as official_save

from expertkit_worker.backends.torch import TorchWeightAdapter
from expertkit_worker.weights.direct_io import AlignedWeightBuffer
from expertkit_worker.weights.disk_cache import WeightDiskCache
from expertkit_worker.weights.dram_cache import DramCache, WeightKey
from expertkit_worker.weights.loader import CachedCpuWeight, CpuWeightLoader
from expertkit_worker.weights.peer_server import PeerWeightServer
from expertkit_worker.weights.transfer import (
    HttpWeightTransfer,
    WeightNotFound,
    WeightTransfer,
    WeightTransferError,
)

_MODEL_NAME = "Qwen/Test Model"
_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 3


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Run one isolated peer-server scenario."""

    return asyncio.run(coroutine)


def make_payload() -> bytes:
    """Return one valid standard expert SafeTensors file."""

    return official_save(
        {
            "model.gate_proj.weight": torch.ones(
                (_INTERMEDIATE_DIM, _HIDDEN_DIM), dtype=torch.float32
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
    """Copy bytes into one aligned application-owned buffer."""

    result = AlignedWeightBuffer(len(payload))
    view = result.view()
    try:
        view[:] = payload
    finally:
        view.release()
    return result


class FakeDiskCache(WeightDiskCache):
    """Serve configured local files with controllable read completion."""

    def __init__(self, results: dict[WeightKey, bytes] | None = None) -> None:
        self.results = {} if results is None else results
        self.reads: list[WeightKey] = []
        self.started = asyncio.Event()
        self.allow = asyncio.Event()
        self.allow.set()

    async def read(self, key: WeightKey, *, max_bytes: int) -> AlignedWeightBuffer:
        self.reads.append(key)
        self.started.set()
        await self.allow.wait()
        try:
            payload = self.results[key]
        except KeyError as error:
            raise FileNotFoundError("expert file does not exist") from error
        if len(payload) > max_bytes:
            raise AssertionError("fixture exceeds loader byte bound")
        return make_buffer(payload)

    async def remove(self, key: WeightKey) -> None:
        self.results.pop(key, None)

    async def write(self, key: WeightKey, buffer: AlignedWeightBuffer) -> None:
        raise NotImplementedError


class UnexpectedTransfer(WeightTransfer):
    """Fail if the peer endpoint tries to fetch a non-local source."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start(self) -> None:
        pass

    async def download(self, url: str, *, max_bytes: int) -> AlignedWeightBuffer:
        self.calls.append(url)
        raise AssertionError("peer serving must not fetch another source")

    async def close(self) -> None:
        pass


def make_loader(
    disk: WeightDiskCache,
    transfer: WeightTransfer,
) -> CpuWeightLoader[object, object]:
    """Return a small CPU loader for the peer-server fixture."""

    adapter = TorchWeightAdapter(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=torch.float32,
        compute_dtype=torch.float32,
        device="cpu",
    )
    cache: DramCache[CachedCpuWeight[object]] = DramCache(
        8 + 16 * 1024 * 1024 + adapter.source_tensor_bytes()
    )
    return CpuWeightLoader(
        model_name=_MODEL_NAME,
        disk_cache=disk,
        weight_server_endpoint="http://unused-weight-server",
        adapter=adapter,
        cache=cache,
        transfer=transfer,
    )


def peer_url(server: PeerWeightServer[object, object], tail: str) -> str:
    """Return one URL using the server's ephemeral localhost port."""

    host, port = server.addresses[0]
    return f"http://{host}:{port}/expert/{tail}"


def test_peer_server_serves_disk_then_dram_without_remote_fetch() -> None:
    async def scenario() -> None:
        key = WeightKey(1, 2)
        payload = make_payload()
        disk = FakeDiskCache({key: payload})
        unexpected = UnexpectedTransfer()
        server: PeerWeightServer[object, object] = PeerWeightServer(
            model_name=_MODEL_NAME,
            num_layers=4,
            experts_per_layer=8,
            host="127.0.0.1",
            port=0,
            max_concurrent_requests=2,
            loader=make_loader(disk, unexpected),
            chunk_bytes=17,
        )
        client = HttpWeightTransfer(max_connections=2, chunk_bytes=13)
        await server.start()
        await client.start()
        try:
            url = peer_url(server, "Qwen%2FTest%20Model/1/2")
            for _ in range(2):
                received = await client.download(url, max_bytes=len(payload))
                view = received.view()
                try:
                    assert bytes(view) == payload
                finally:
                    view.release()
                    received.close()
            assert disk.reads == [key]
            assert unexpected.calls == []
        finally:
            await client.close()
            await server.close()

    run(scenario())


def test_peer_server_returns_not_found_without_recursive_fetch() -> None:
    async def scenario() -> None:
        unexpected = UnexpectedTransfer()
        server: PeerWeightServer[object, object] = PeerWeightServer(
            model_name=_MODEL_NAME,
            num_layers=4,
            experts_per_layer=8,
            host="127.0.0.1",
            port=0,
            max_concurrent_requests=1,
            loader=make_loader(FakeDiskCache(), unexpected),
        )
        client = HttpWeightTransfer(max_connections=1)
        await server.start()
        await client.start()
        try:
            with pytest.raises(WeightNotFound):
                await client.download(
                    peer_url(server, "Qwen%2FTest%20Model/1/2"),
                    max_bytes=1024,
                )
            assert unexpected.calls == []
        finally:
            await client.close()
            await server.close()

    run(scenario())


def test_peer_server_returns_busy_when_request_limit_is_full() -> None:
    async def scenario() -> None:
        key = WeightKey(1, 2)
        payload = make_payload()
        disk = FakeDiskCache({key: payload})
        disk.allow.clear()
        server: PeerWeightServer[object, object] = PeerWeightServer(
            model_name=_MODEL_NAME,
            num_layers=4,
            experts_per_layer=8,
            host="127.0.0.1",
            port=0,
            max_concurrent_requests=1,
            loader=make_loader(disk, UnexpectedTransfer()),
        )
        client = HttpWeightTransfer(max_connections=2)
        await server.start()
        await client.start()
        try:
            url = peer_url(server, "Qwen%2FTest%20Model/1/2")
            first = asyncio.create_task(client.download(url, max_bytes=len(payload)))
            await asyncio.wait_for(disk.started.wait(), timeout=1)
            with pytest.raises(WeightTransferError) as caught:
                await client.download(url, max_bytes=len(payload))
            assert caught.value.retryable is True
            disk.allow.set()
            received = await asyncio.wait_for(first, timeout=1)
            received.close()
        finally:
            disk.allow.set()
            await client.close()
            await server.close()

    run(scenario())
