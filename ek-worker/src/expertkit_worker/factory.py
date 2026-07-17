"""Construct one configured Worker process from concrete MVP components."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import structlog
import torch
from expertkit_transport.adapters.grpc import GrpcBatchSpec, GrpcWorkerServer
from expertkit_transport.contracts import WorkerPositionSpec

from expertkit_worker.app import WorkerApplication
from expertkit_worker.backends import ComputeBackend
from expertkit_worker.config import (
    ActivationDType,
    BackendName,
    WorkerConfig,
    plan_device_resources,
    validate_available_device_memory,
)
from expertkit_worker.control import (
    ControllerConnection,
    ControllerSupervisor,
    ExpertStateReporter,
    HeartbeatSender,
    WeightControlSession,
    WorkerRegistration,
    new_start_id,
)
from expertkit_worker.execution import WorkerExecution
from expertkit_worker.observability import create_observability
from expertkit_worker.weights import (
    CpuWeightLoader,
    DirectIOWeightDiskCache,
    DiskWriteback,
    ExpertStateChange,
    PeerWeightServer,
    WeightManager,
    max_safetensors_file_bytes,
)
from expertkit_worker.weights.adapter import WeightAdapter
from expertkit_worker.weights.dram_cache import DramCache
from expertkit_worker.weights.loader import CachedCpuWeight
from expertkit_worker.weights.transfer import HttpWeightTransfer

logger = structlog.get_logger(__name__)

_DTYPE = {
    ActivationDType.FP16: torch.float16,
    ActivationDType.BF16: torch.bfloat16,
    ActivationDType.FP32: torch.float32,
}


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


def _create_weight_adapter(
    config: WorkerConfig,
    *,
    source_dtype: torch.dtype,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> WeightAdapter[Any, Any]:
    if config.worker.backend is BackendName.TORCH:
        from expertkit_worker.backends.torch import TorchWeightAdapter

        return TorchWeightAdapter(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            source_dtype=source_dtype,
            compute_dtype=compute_dtype,
            device=device,
        )
    if config.worker.backend is BackendName.GGML:
        try:
            from expertkit_worker.backends.ggml import GgmlWeightAdapter
        except ModuleNotFoundError as error:
            if error.name == "ggml":
                raise RuntimeError("the GGML Backend requires the locked ggml extra") from error
            raise

        return GgmlWeightAdapter(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            source_dtype=source_dtype,
            compute_dtype=compute_dtype,
        )
    raise NotImplementedError("the fused Backend is not implemented in this build")


def _create_backend(
    config: WorkerConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
    acquire_many: Callable[[int, tuple[int, ...]], Any],
) -> ComputeBackend:
    if config.worker.backend is BackendName.TORCH:
        from expertkit_worker.backends.torch import TorchBackend

        return TorchBackend(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            top_k=config.model.top_k,
            dtype=dtype,
            device=device,
            acquire_many=acquire_many,
        )
    if config.worker.backend is BackendName.GGML:
        from expertkit_worker.backends.ggml import GgmlBackend

        if config.ggml is None:
            raise ValueError("ggml configuration is missing after validation")
        return GgmlBackend(
            hidden_dim=config.model.hidden_dim,
            intermediate_dim=config.model.expert_intermediate_dim,
            top_k=config.model.top_k,
            dtype=dtype,
            cpu_threads=config.ggml.cpu_threads,
            acquire_many=acquire_many,
        )
    raise NotImplementedError("the fused Backend is not implemented in this build")


def _memory_info(device: torch.device) -> tuple[int, int]:
    if device.type == "cuda":
        available, total = torch.cuda.mem_get_info(device)
        return int(available), int(total)
    if device.type != "cpu":
        raise ValueError("Worker device must be CPU or CUDA")
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot query available CPU memory") from error
    if min(page_size, available_pages, total_pages) <= 0:
        raise RuntimeError("the operating system returned invalid CPU memory information")
    return int(available_pages * page_size), int(total_pages * page_size)


async def build_worker_application(config: WorkerConfig) -> WorkerApplication:
    """Build the selected MVP Worker without starting network listeners.

    Raises:
        NotImplementedError: The fused Backend remains disabled in this build.
        RuntimeError: A selected Backend extra is absent or memory cannot be queried.
        ValueError: Startup resource planning or a component contract is invalid.
    """

    if not isinstance(config, WorkerConfig):
        raise TypeError("config must be a WorkerConfig")
    activation_dtype = _DTYPE[config.model.activation_dtype]
    weight_dtype = _DTYPE[config.model.weight_dtype]
    device = torch.device(config.worker.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    observability = create_observability(
        config.observability,
        worker_id=config.worker.id,
    )
    metrics = observability.metrics
    receiver: GrpcWorkerServer | None = None
    execution: WorkerExecution | None = None
    manager: WeightManager[Any, Any] | None = None
    peer_server: PeerWeightServer[Any, Any] | None = None
    transfer: HttpWeightTransfer | None = None
    disk_cache: DirectIOWeightDiskCache | None = None
    try:
        adapter = _create_weight_adapter(
            config,
            source_dtype=weight_dtype,
            compute_dtype=activation_dtype,
            device=device,
        )
        batch_spec = GrpcBatchSpec(
            instance_id=config.model.instance_id,
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            max_batch_tokens=config.worker.max_batch_tokens,
            hidden_dim=config.model.hidden_dim,
            top_k=config.model.top_k,
            dtype=activation_dtype,
        )
        receiver = GrpcWorkerServer(
            config.transport.grpc.listen,
            batch_spec,
            max_active_batches=config.worker.max_active_batches_per_device,
            max_pending_batches=config.transport.max_pending_batches_per_device,
            interceptors=observability.grpc_interceptors,
            on_rejection=metrics.batch_rejected,
            on_pending_changed=metrics.pending_batches_changed,
        )

        manager_holder: list[WeightManager[Any, Any] | None] = [None]

        def acquire_many(layer_id: int, expert_ids: tuple[int, ...]) -> Any:
            current = manager_holder[0]
            if current is None:
                raise RuntimeError("Weight Manager is not installed in the selected Backend")
            return current.acquire_many(layer_id, expert_ids)

        backend = _create_backend(
            config,
            dtype=activation_dtype,
            device=device,
            acquire_many=acquire_many,
        )
        position_spec = WorkerPositionSpec(
            max_batch_tokens=config.worker.max_batch_tokens,
            hidden_dim=config.model.hidden_dim,
            top_k=config.model.top_k,
            dtype=activation_dtype,
            device=device,
        )
        execution = WorkerExecution(
            receiver,
            backend,
            instance_id=config.model.instance_id,
            position_spec=position_spec,
            active_positions=config.worker.max_active_batches_per_device,
            metrics=metrics,
        )
        resource_plan = plan_device_resources(
            device_memory_limit_bytes=int(config.worker.device_memory_limit),
            fixed_position_bytes=execution.fixed_device_bytes,
            backend_estimate=backend.estimate_resources(config.worker.max_batch_tokens),
            active_batches=config.worker.max_active_batches_per_device,
            conversion_temporary_bytes=adapter.conversion_temporary_bytes(),
        )
        available_bytes, total_bytes = _memory_info(device)
        if int(config.worker.device_memory_limit) > total_bytes:
            raise ValueError("worker.device_memory_limit exceeds total device memory")
        validate_available_device_memory(
            resource_plan,
            available_bytes_after_fixed_positions=available_bytes,
        )

        disk_cache = DirectIOWeightDiskCache(
            root=config.weight_manager.disk_cache.path,
            model_name=config.model.name,
            max_concurrent_operations=config.weight_manager.max_concurrent_loads,
        )
        transfer = HttpWeightTransfer(
            max_connections=config.weight_manager.max_concurrent_loads,
        )
        dram_cache: DramCache[CachedCpuWeight[Any]] = DramCache(_dram_cache_limit(config, adapter))
        loader = CpuWeightLoader(
            model_name=config.model.name,
            disk_cache=disk_cache,
            weight_server_endpoint=str(config.weight_manager.weight_server_endpoint),
            adapter=adapter,
            cache=dram_cache,
            transfer=transfer,
            source_result=lambda source, success: metrics.weight_source_result(
                source,
                success=success,
            ),
        )
        writeback = DiskWriteback(
            disk_cache=disk_cache,
            enabled=config.weight_manager.disk_cache.writeback,
            max_pending=config.weight_manager.max_concurrent_loads,
        )
        reporter = ExpertStateReporter(
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            max_updates=config.weight_manager.state_report.max_updates,
            max_delay_ms=config.weight_manager.state_report.max_delay_ms,
        )

        def state_changed(change: ExpertStateChange) -> None:
            reporter.record(change)
            metrics.expert_state_changed(change.expert.state.value)

        manager = WeightManager(
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            device=config.worker.device,
            device_weight_capacity_bytes=resource_plan.weight_capacity_bytes,
            max_concurrent_loads=config.weight_manager.max_concurrent_loads,
            adapter=adapter,
            loader=loader,
            writeback=writeback,
            state_changed=state_changed,
            device_bytes_changed=metrics.device_weight_bytes_changed,
        )
        manager_holder[0] = manager

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

        start_id = new_start_id()
        connection = ControllerConnection(config.controller.endpoint)
        heartbeat = HeartbeatSender(
            worker_id=config.worker.id,
            start_id=start_id,
            interval_secs=config.controller.heartbeat_interval_secs,
        )
        weights = WeightControlSession(
            worker_id=config.worker.id,
            start_id=start_id,
            max_experts=manager.max_experts,
            shutdown_grace_secs=config.worker.shutdown_grace_secs,
            manager=manager,
            reporter=reporter,
            receiver=receiver,
        )
        registration = WorkerRegistration(
            worker_id=config.worker.id,
            start_id=start_id,
            instance_id=config.model.instance_id,
            computation_endpoint=config.transport.grpc.advertise,
            peer_weight_endpoint=str(config.weight_manager.peer.advertise),
            backend=config.worker.backend.value,
            activation_dtype=activation_dtype,
            device=config.worker.device,
            max_experts=manager.max_experts,
            max_batch_tokens=config.worker.max_batch_tokens,
            max_active_batches=config.worker.max_active_batches_per_device,
            max_pending_batches=config.transport.max_pending_batches_per_device,
        )
        control = ControllerSupervisor(
            connection=connection,
            registration=registration,
            heartbeat=heartbeat,
            weights=weights,
            registration_timeout_secs=config.controller.heartbeat_timeout_secs,
            stable_stream_secs=config.controller.heartbeat_timeout_secs,
        )
        logger.info(
            "worker_resource_plan",
            device=config.worker.device,
            device_total_bytes=total_bytes,
            device_available_bytes=available_bytes,
            device_memory_limit_bytes=resource_plan.device_memory_limit_bytes,
            runtime_reserve_bytes=resource_plan.runtime_reserve_bytes,
            weight_capacity_bytes=resource_plan.weight_capacity_bytes,
            max_experts=manager.max_experts,
            fixed_host_staging_bytes=execution.fixed_host_staging_bytes,
        )
        return WorkerApplication(
            transfer=transfer,
            manager=manager,
            peer_server=peer_server,
            computation_server=receiver,
            execution=execution,
            control=control,
            disk_cache=disk_cache,
            observability=observability,
        )
    except BaseException:
        if execution is not None:
            await execution.close()
        elif receiver is not None:
            await receiver.close()
        if peer_server is not None:
            await peer_server.close()
        if manager is not None:
            await manager.close()
        if transfer is not None:
            await transfer.close()
        if disk_cache is not None:
            disk_cache.close()
        await observability.close()
        raise
