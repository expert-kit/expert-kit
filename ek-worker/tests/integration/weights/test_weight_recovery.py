"""Integration coverage for Weight Server writeback and offline disk recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import torch
from aiohttp import web
from safetensors.torch import save as official_save

from expertkit_worker.backends import BackendBatch
from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights, TorchWeightAdapter
from expertkit_worker.device import CpuWorkerRuntime
from expertkit_worker.weights import (
    CachedCpuWeight,
    CpuWeightLoader,
    DirectIOWeightDiskCache,
    DiskWriteback,
    ExpertStateKind,
    TargetExpert,
    WeightManager,
    max_safetensors_file_bytes,
)
from expertkit_worker.weights.dram_cache import DramCache
from expertkit_worker.weights.transfer import HttpWeightTransfer

_MODEL_NAME = "Qwen/Test Model"
_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 3


def _payload() -> bytes:
    return official_save(
        {
            "model.expert.gate_proj.weight": torch.tensor(
                [
                    [0.2, -0.1, 0.3, 0.4],
                    [-0.3, 0.5, 0.1, -0.2],
                    [0.6, 0.2, -0.4, 0.1],
                ],
                dtype=torch.float32,
            ),
            "model.expert.up_proj.weight": torch.tensor(
                [
                    [0.4, 0.1, -0.2, 0.3],
                    [0.2, -0.5, 0.6, 0.1],
                    [-0.1, 0.3, 0.2, 0.5],
                ],
                dtype=torch.float32,
            ),
            "model.expert.down_proj.weight": torch.tensor(
                [
                    [0.2, 0.4, -0.3],
                    [-0.1, 0.5, 0.2],
                    [0.3, -0.2, 0.6],
                    [0.4, 0.1, -0.5],
                ],
                dtype=torch.float32,
            ),
        }
    )


async def _with_weight_server(
    payload: bytes,
    scenario: Callable[[str, list[str]], Awaitable[None]],
) -> None:
    requests: list[str] = []

    async def serve(request: web.Request) -> web.Response:
        requests.append(request.path)
        return web.Response(body=payload)

    app = web.Application()
    app.router.add_get("/expert/{tail:.*}", serve)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        await scenario(f"http://127.0.0.1:{port}", requests)
    finally:
        await runner.cleanup()


def _adapter(runtime: CpuWorkerRuntime) -> TorchWeightAdapter:
    return TorchWeightAdapter(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        source_dtype=torch.float32,
        compute_dtype=torch.float32,
        runtime=runtime,
    )


def _manager(
    root: Path,
    endpoint: str,
    runtime: CpuWorkerRuntime,
) -> tuple[
    WeightManager[TorchExpertWeights, TorchExpertWeights],
    DirectIOWeightDiskCache,
    HttpWeightTransfer,
]:
    adapter = _adapter(runtime)
    disk = DirectIOWeightDiskCache(
        root=root,
        model_name=_MODEL_NAME,
        max_concurrent_operations=1,
    )
    transfer = HttpWeightTransfer(max_connections=1)
    cache: DramCache[CachedCpuWeight[TorchExpertWeights]] = DramCache(
        max_safetensors_file_bytes(adapter.source_tensor_bytes())
    )
    loader = CpuWeightLoader(
        model_name=_MODEL_NAME,
        disk_cache=disk,
        weight_server_endpoint=endpoint,
        adapter=adapter,
        cache=cache,
        transfer=transfer,
    )
    manager = WeightManager(
        num_layers=1,
        experts_per_layer=1,
        device="cpu",
        device_weight_capacity_bytes=adapter.ready_weight_bytes(),
        max_concurrent_loads=1,
        adapter=adapter,
        loader=loader,
        writeback=DiskWriteback(
            disk_cache=disk,
            enabled=True,
            max_pending=1,
        ),
    )
    return manager, disk, transfer


def _compute(
    manager: WeightManager[TorchExpertWeights, TorchExpertWeights],
    runtime: CpuWorkerRuntime,
) -> torch.Tensor:
    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=1,
        dtype=torch.float32,
        runtime=runtime,
        acquire_many=manager.acquire_many,
    )
    hidden_states = torch.tensor(
        [[0.2, -0.4, 0.8, 0.5], [-0.1, 0.7, 0.3, -0.6]],
        dtype=torch.float32,
    )
    batch = BackendBatch(
        layer_id=0,
        hidden_states=hidden_states,
        expert_ids=torch.zeros((2, 1), dtype=torch.int32),
        routing_weights=torch.tensor([[0.25], [0.8]], dtype=torch.float32),
        distinct_expert_ids=(0,),
    )
    output = torch.empty_like(hidden_states)
    completion = backend.submit(batch, output)
    completion.wait_host()
    completion.close()
    return output


async def _load_and_compute(
    root: Path,
    endpoint: str,
) -> tuple[torch.Tensor, Path]:
    runtime = CpuWorkerRuntime(torch.device("cpu"))
    manager, disk, transfer = _manager(root, endpoint, runtime)
    await transfer.start()
    manager.start()
    try:
        await manager.apply_targets(1, [TargetExpert(0, 0, "cpu")])
        async with asyncio.timeout(5):
            await manager.wait_for_idle()
        generation, states = await manager.snapshot()
        assert generation == 1
        assert len(states) == 1
        assert states[0].state is ExpertStateKind.READY
        return _compute(manager, runtime), disk.path(TargetExpert(0, 0, "cpu").key)
    finally:
        await manager.close()
        await transfer.close()
        disk.close()


def test_weight_server_writeback_supports_offline_restart(tmp_path: Path) -> None:
    async def scenario(endpoint: str, requests: list[str]) -> None:
        online_output, cache_path = await _load_and_compute(tmp_path, endpoint)

        assert requests == ["/expert/Qwen/Test Model/0/0"]
        assert cache_path.is_file()

        unreachable_endpoint = "http://127.0.0.1:1"
        offline_output, same_cache_path = await _load_and_compute(
            tmp_path,
            unreachable_endpoint,
        )

        assert same_cache_path == cache_path
        torch.testing.assert_close(offline_output, online_output)
        assert requests == ["/expert/Qwen/Test Model/0/0"]

    asyncio.run(_with_weight_server(_payload(), scenario))
