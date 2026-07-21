"""Construct the Weight Manager and its loading services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from expertkit_worker.config import WorkerConfig
from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.disk_cache import DirectIOWeightDiskCache
from expertkit_worker.weights.dram_cache import DramCache
from expertkit_worker.weights.format import max_safetensors_file_bytes
from expertkit_worker.weights.loader import CachedCpuWeight, CpuWeightLoader
from expertkit_worker.weights.manager import ExpertStateChange, WeightManager
from expertkit_worker.weights.peer_server import PeerWeightServer
from expertkit_worker.weights.transfer import HttpWeightTransfer
from expertkit_worker.weights.writeback import DiskWriteback


@dataclass(frozen=True, slots=True)
class WeightServices:
    """Hold the concrete weight components transferred to `WorkerApplication`."""

    disk_cache: DirectIOWeightDiskCache
    transfer: HttpWeightTransfer
    manager: WeightManager[Any, Any]
    peer_server: PeerWeightServer[Any, Any]

    async def close(self) -> None:
        """Close all components after a later startup step fails."""

        await self.peer_server.close()
        await self.manager.close()
        await self.transfer.close()
        self.disk_cache.close()


def _split_address(value: str) -> tuple[str, int]:
    if value.startswith("["):
        closing = value.index("]")
        return value[1:closing], int(value[closing + 2 :])
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _dram_cache_limit(config: WorkerConfig, adapter: WeightAdapter[Any, Any]) -> int:
    one_entry = (
        max_safetensors_file_bytes(adapter.source_tensor_bytes()) + adapter.cpu_extra_bytes()
    )
    configured = config.weight_manager.dram_cache.max_bytes
    if configured is not None:
        resolved = int(configured)
        if resolved < one_entry:
            raise ValueError("weight_manager.dram_cache.max_bytes cannot fit one expert")
        return resolved
    expert_count = config.model.num_layers * config.model.experts_per_layer
    return one_entry * expert_count


async def create_weight_services(
    config: WorkerConfig,
    *,
    disk_cache: DirectIOWeightDiskCache,
    adapter: WeightAdapter[Any, Any],
    device_weight_capacity_bytes: int,
    state_changed: Callable[[ExpertStateChange], None],
    source_result: Callable[[str, bool], None],
    device_bytes_changed: Callable[[int], None],
) -> WeightServices:
    """Build weight loading, caching, placement, and peer-serving components.

    The function closes every component it has created if construction fails.
    On success, ownership transfers to the returned `WeightServices` value.
    """

    transfer: HttpWeightTransfer | None = None
    manager: WeightManager[Any, Any] | None = None
    peer_server: PeerWeightServer[Any, Any] | None = None
    try:
        transfer = HttpWeightTransfer(
            max_connections=config.weight_manager.max_concurrent_loads,
        )
        cache: DramCache[CachedCpuWeight[Any]] = DramCache(_dram_cache_limit(config, adapter))
        loader = CpuWeightLoader(
            model_name=config.model.name,
            disk_cache=disk_cache,
            weight_server_endpoint=str(config.weight_manager.weight_server_endpoint),
            adapter=adapter,
            cache=cache,
            transfer=transfer,
            source_result=lambda source, success: source_result(source, success),
        )
        writeback = DiskWriteback(
            disk_cache=disk_cache,
            enabled=config.weight_manager.disk_cache.writeback,
            max_pending=config.weight_manager.max_concurrent_loads,
        )
        manager = WeightManager(
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            device=config.worker.device,
            device_weight_capacity_bytes=device_weight_capacity_bytes,
            max_concurrent_loads=config.weight_manager.max_concurrent_loads,
            adapter=adapter,
            loader=loader,
            writeback=writeback,
            state_changed=state_changed,
            device_bytes_changed=device_bytes_changed,
        )
        peer_host, peer_port = _split_address(config.weight_manager.peer.listen)
        peer_server = PeerWeightServer(
            model_name=config.model.name,
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            host=peer_host,
            port=peer_port,
            max_concurrent_requests=config.weight_manager.max_concurrent_loads,
            loader=loader,
        )
        return WeightServices(
            disk_cache=disk_cache,
            transfer=transfer,
            manager=manager,
            peer_server=peer_server,
        )
    except BaseException:
        if peer_server is not None:
            await peer_server.close()
        if manager is not None:
            await manager.close()
        if transfer is not None:
            await transfer.close()
        disk_cache.close()
        raise


async def create_weight_disk_cache(config: WorkerConfig) -> DirectIOWeightDiskCache:
    """Create and probe the configured direct-I/O cache before device allocation."""

    disk_cache = DirectIOWeightDiskCache(
        root=Path(config.weight_manager.disk_cache.path),
        model_name=config.model.name,
        max_concurrent_operations=config.weight_manager.max_concurrent_loads,
    )
    try:
        await disk_cache.initialize()
    except BaseException:
        disk_cache.close()
        raise
    return disk_cache
