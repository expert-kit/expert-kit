"""Construct one configured Worker process from concrete MVP components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import structlog
import torch
from expertkit_transport.controller import (
    ResolvedDefaultInstance,
    resolve_default_instance,
)
from expertkit_transport.transports import WorkerBatchBuffers
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    WorkerBatchReceiver,
    WorkerEndpointConfig,
)
from expertkit_transport.transports.grpc import GrpcWorkerBatchReceiver
from expertkit_transport.transports.shm import ShmWorkerBatchReceiver

from expertkit_worker.app import WorkerApplication
from expertkit_worker.backends.factory import (
    create_compute_backend,
    create_weight_adapter,
    torch_dtype,
)
from expertkit_worker.config import (
    GrpcTransportConfig,
    ShmTransportConfig,
    WorkerConfig,
    plan_device_resources,
    validate_available_device_memory,
)
from expertkit_worker.control import (
    ExpertStateReporter,
)
from expertkit_worker.control.factory import create_controller_supervisor
from expertkit_worker.device import (
    AsyncWorkerDeviceRuntime,
    CpuWorkerRuntime,
    CudaWorkerRuntime,
    WorkerDeviceRuntime,
)
from expertkit_worker.execution import (
    AsyncExecutionSlot,
    CpuExecutionSlot,
    ExecutionSlot,
    ExecutionSlotFactory,
    WorkerExecutor,
)
from expertkit_worker.observability import create_observability
from expertkit_worker.weights import (
    DirectIOWeightDiskCache,
    ExpertStateChange,
    WeightManager,
)
from expertkit_worker.weights.factory import (
    WeightServices,
    create_weight_disk_cache,
    create_weight_services,
)

logger = structlog.get_logger(__name__)


class _InstanceResolver(Protocol):
    async def __call__(
        self,
        controller_endpoint: str,
        *,
        requested_instance_id: int | None,
        timeout_seconds: float,
    ) -> ResolvedDefaultInstance: ...


# NOTE: This composition is needed because we might want to create
# ExecutionSlot with AsyncWorkerDeviceRuntime, but for others
# only the WorkerDeviceRuntime is used
@dataclass(frozen=True, slots=True)
class _DeviceWiring:
    runtime: WorkerDeviceRuntime
    create_slot: ExecutionSlotFactory


def _create_device_wiring(device_name: str) -> _DeviceWiring:
    if device_name == "cpu":
        return _cpu_wiring(CpuWorkerRuntime(torch.device(device_name)))

    if device_name.startswith("cuda:"):
        return _async_wiring(CudaWorkerRuntime(torch.device(device_name)))

    if device_name.startswith("npu:"):
        # Lazy import AscendWorkerRuntime which includes torch_npu
        from expertkit_worker.device.ascend import AscendWorkerRuntime

        return _async_wiring(AscendWorkerRuntime(torch.device(device_name)))

    raise NotImplementedError(f"{device_name} is not supported.")


# Two wirings for cpu or async execution slot
def _cpu_wiring(runtime: WorkerDeviceRuntime) -> _DeviceWiring:
    def create_slot(
        spec: BatchBufferConfig,
        transport_buffers: WorkerBatchBuffers,
        *,
        enable_device_timing: bool = False,
    ) -> ExecutionSlot:
        return CpuExecutionSlot(spec, transport_buffers, runtime=runtime)

    return _DeviceWiring(runtime=runtime, create_slot=create_slot)


def _async_wiring[StreamT, EventT](
    runtime: AsyncWorkerDeviceRuntime[StreamT, EventT],
) -> _DeviceWiring:
    def create_slot(
        spec: BatchBufferConfig,
        transport_buffers: WorkerBatchBuffers,
        *,
        enable_device_timing: bool = False,
    ) -> ExecutionSlot:
        return AsyncExecutionSlot(
            spec,
            transport_buffers,
            enable_device_timing=enable_device_timing,
            runtime=runtime,
        )

    runtime.set_current_device()
    return _DeviceWiring(runtime=runtime, create_slot=create_slot)


async def build_worker_application(
    config: WorkerConfig,
    *,
    instance_resolver: _InstanceResolver = resolve_default_instance,
) -> WorkerApplication:
    """Build the selected MVP Worker without starting network listeners.

    Args:
        instance_resolver: Startup-only Controller resolver. Tests may replace
            it to avoid opening a control channel.

    Raises:
        RuntimeError: A selected Backend extra is absent or memory cannot be queried.
        ValueError: Startup resource planning or a component contract is invalid.
    """

    if not isinstance(config, WorkerConfig):
        raise TypeError("config must be a WorkerConfig")
    resolved_instance = await instance_resolver(
        config.controller.endpoint,
        requested_instance_id=config.model.instance_id,
        timeout_seconds=config.controller.heartbeat_timeout_secs,
    )
    instance_id = resolved_instance.instance_id
    activation_dtype = torch_dtype(config.model.activation_dtype)
    weight_dtype = torch_dtype(config.model.weight_dtype)

    device_wiring = _create_device_wiring(config.worker.device)
    runtime = device_wiring.runtime
    create_slot = device_wiring.create_slot
    device = runtime.device

    observability = create_observability(
        config.observability,
        worker_id=config.worker.id,
    )
    metrics = observability.metrics
    receiver: WorkerBatchReceiver | None = None
    execution: WorkerExecutor | None = None
    disk_cache: DirectIOWeightDiskCache | None = None
    weight_services: WeightServices | None = None
    try:
        disk_cache = await create_weight_disk_cache(config)
        adapter = create_weight_adapter(
            config,
            source_dtype=weight_dtype,
            compute_dtype=activation_dtype,
            runtime=runtime,
        )
        endpoint_config = WorkerEndpointConfig(
            instance_id=instance_id,
            num_layers=config.model.num_layers,
            experts_per_layer=config.model.experts_per_layer,
            max_batch_tokens=config.worker.max_batch_tokens,
            hidden_dim=config.model.hidden_dim,
            top_k=config.model.top_k,
            dtype=activation_dtype,
        )
        receiver_options = {
            "max_active_batches": config.worker.max_active_batches_per_device,
            "max_pending_batches": config.transport.max_pending_batches_per_device,
            "interceptors": observability.grpc_interceptors,
            "tracer": observability.tracer,
            "on_rejection": metrics.batch_rejected,
            "on_pending_changed": metrics.pending_batches_changed,
        }
        if isinstance(config.transport, GrpcTransportConfig):
            receiver = GrpcWorkerBatchReceiver(
                config.transport.listen,
                endpoint_config,
                **receiver_options,
            )
            computation_endpoint = config.transport.advertise
            transport_type = "grpc"
        elif isinstance(config.transport, ShmTransportConfig):
            receiver = ShmWorkerBatchReceiver(
                config.transport.rpc_listen,
                endpoint_config,
                shared_memory_dir=Path(config.transport.shared_memory_dir),
                **receiver_options,
            )
            computation_endpoint = config.transport.rpc_advertise
            transport_type = "shm"
        else:
            raise AssertionError("validated Worker configuration selected no Transport")

        manager_holder: list[WeightManager[Any, Any] | None] = [None]

        def acquire_many(layer_id: int, expert_ids: tuple[int, ...]) -> Any:
            current = manager_holder[0]
            if current is None:
                raise RuntimeError("Weight Manager is not installed in the selected Backend")
            return current.acquire_many(layer_id, expert_ids)

        backend = create_compute_backend(
            config,
            dtype=activation_dtype,
            runtime=runtime,
            acquire_many=acquire_many,
        )
        buffer_config = BatchBufferConfig(
            max_batch_tokens=config.worker.max_batch_tokens,
            hidden_dim=config.model.hidden_dim,
            top_k=config.model.top_k,
            dtype=activation_dtype,
            device=device,
        )
        execution = WorkerExecutor(
            receiver,
            backend,
            create_slot=create_slot,
            instance_id=instance_id,
            buffer_config=buffer_config,
            slot_count=config.worker.max_active_batches_per_device,
            metrics=metrics,
            tracer=observability.tracer,
        )
        resource_plan = plan_device_resources(
            device_memory_limit_bytes=int(config.worker.device_memory_limit),
            fixed_slot_bytes=execution.fixed_device_bytes,
            backend_estimate=backend.estimate_resources(config.worker.max_batch_tokens),
            active_batches=config.worker.max_active_batches_per_device,
            conversion_temporary_bytes=adapter.conversion_temporary_bytes(),
        )
        available_bytes, total_bytes = runtime.memory_info()
        if int(config.worker.device_memory_limit) > total_bytes:
            raise ValueError("worker.device_memory_limit exceeds total device memory")
        validate_available_device_memory(
            resource_plan,
            available_bytes_after_fixed_slots=available_bytes,
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

        weight_services = await create_weight_services(
            config,
            disk_cache=disk_cache,
            adapter=adapter,
            device_weight_capacity_bytes=resource_plan.weight_capacity_bytes,
            state_changed=state_changed,
            source_result=lambda source, success: metrics.weight_source_result(
                source,
                success=success,
            ),
            device_bytes_changed=metrics.device_weight_bytes_changed,
        )
        disk_cache = weight_services.disk_cache
        transfer = weight_services.transfer
        manager = weight_services.manager
        peer_server = weight_services.peer_server
        manager_holder[0] = manager

        control = create_controller_supervisor(
            config,
            instance_id=instance_id,
            activation_dtype=activation_dtype,
            receiver=receiver,
            manager=manager,
            reporter=reporter,
            computation_endpoint=computation_endpoint,
            transport_type=transport_type,
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
        if weight_services is not None:
            await weight_services.close()
        elif disk_cache is not None:
            disk_cache.close()
        await observability.close()
        raise
