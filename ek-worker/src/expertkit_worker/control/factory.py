"""Construct Controller registration, heartbeat, and weight-control services."""

from __future__ import annotations

import torch
from expertkit_transport.transports.base import WorkerBatchReceiver

from expertkit_worker.config import WorkerConfig
from expertkit_worker.control.lifecycle import (
    ControllerConnection,
    HeartbeatSender,
    WorkerRegistration,
    WorkerRuntimeIdentity,
)
from expertkit_worker.control.state_reporter import ExpertStateReporter
from expertkit_worker.control.supervisor import ControllerSupervisor
from expertkit_worker.control.weight_stream import WeightControlSession
from expertkit_worker.weights.manager import WeightManager


def create_controller_supervisor(
    config: WorkerConfig,
    *,
    identity: WorkerRuntimeIdentity,
    instance_id: int,
    activation_dtype: torch.dtype,
    receiver: WorkerBatchReceiver,
    manager: WeightManager[object, object],
    reporter: ExpertStateReporter,
    computation_endpoint: str,
    transport_type: str,
) -> ControllerSupervisor:
    """Create the complete Controller-facing control path for one Worker."""

    connection = ControllerConnection(config.controller.endpoint)
    heartbeat = HeartbeatSender(
        worker_id=identity.worker_id,
        start_id=identity.start_id,
        interval_secs=config.controller.heartbeat_interval_secs,
    )
    weights = WeightControlSession(
        worker_id=identity.worker_id,
        start_id=identity.start_id,
        max_experts=manager.max_experts,
        shutdown_grace_secs=config.worker.shutdown_grace_secs,
        manager=manager,
        reporter=reporter,
        receiver=receiver,
    )
    registration = WorkerRegistration(
        worker_id=identity.worker_id,
        start_id=identity.start_id,
        instance_id=instance_id,
        computation_endpoint=computation_endpoint,
        peer_weight_endpoint=str(config.weight_manager.peer.advertise),
        backend=config.worker.backend.value,
        activation_dtype=activation_dtype,
        device=config.worker.device,
        max_experts=manager.max_experts,
        max_batch_tokens=config.worker.max_batch_tokens,
        max_active_batches=config.worker.max_active_batches_per_device,
        max_pending_batches=config.transport.max_pending_batches_per_device,
        transport_type=transport_type,
    )
    return ControllerSupervisor(
        connection=connection,
        registration=registration,
        heartbeat=heartbeat,
        weights=weights,
        registration_timeout_secs=config.controller.heartbeat_timeout_secs,
        stable_stream_secs=config.controller.heartbeat_timeout_secs,
    )
