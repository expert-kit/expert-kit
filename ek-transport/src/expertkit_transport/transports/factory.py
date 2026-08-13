"""Create Frontend connections for Worker Transport types published by Controller."""

from __future__ import annotations

from typing import cast

import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.transports.base import (
    WorkerEndpointConfig,
    WorkerTransport,
    WorkerTransportRuntimeRegistry,
)
from expertkit_transport.transports.grpc.client import GrpcWorkerTransport
from expertkit_transport.transports.nccl import (
    NcclRuntimeProtocol,
    NcclWorkerTransport,
)
from expertkit_transport.transports.shm.client import ShmWorkerTransport


def _create_transfer_engine_transport(
    endpoint: str,
    endpoint_config: WorkerEndpointConfig,
    *,
    max_in_flight: int,
    device: torch.device,
    runtime: object,
    worker_start_id: str,
) -> WorkerTransport:
    """Import the optional Mooncake-backed implementation only when selected."""

    from expertkit_transport.transports.transfer_engine import (
        TransferEngineRuntimeProtocol,
        TransferEngineWorkerTransport,
    )

    if not isinstance(runtime, TransferEngineRuntimeProtocol):
        raise TypeError("Transfer Engine Worker transport received an incompatible runtime")
    return TransferEngineWorkerTransport(
        endpoint,
        endpoint_config,
        max_in_flight=max_in_flight,
        device=device,
        runtime=runtime,
        expected_worker_start_id=worker_start_id,
    )


def create_worker_transport(
    *,
    transport_type: int,
    endpoint: str,
    instance_id: int,
    num_layers: int,
    experts_per_layer: int,
    max_batch_tokens: int,
    hidden_dim: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
    max_in_flight: int,
    worker_start_id: str,
    runtime_registry: WorkerTransportRuntimeRegistry | None = None,
) -> WorkerTransport:
    """Create the data connection declared by one validated Worker route."""

    endpoint_config = WorkerEndpointConfig(
        instance_id=instance_id,
        num_layers=num_layers,
        experts_per_layer=experts_per_layer,
        max_batch_tokens=max_batch_tokens,
        hidden_dim=hidden_dim,
        top_k=top_k,
        dtype=dtype,
    )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_GRPC:
        return GrpcWorkerTransport(
            endpoint,
            endpoint_config,
            max_in_flight=max_in_flight,
            device=device,
        )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_SHM:
        return ShmWorkerTransport(
            endpoint,
            endpoint_config,
            max_in_flight=max_in_flight,
            device=device,
        )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_NCCL:
        runtime = None if runtime_registry is None else runtime_registry.runtime_for(transport_type)
        if runtime is None:
            raise ValueError("NCCL Worker transport requires a process-level runtime")
        return NcclWorkerTransport(
            endpoint,
            endpoint_config,
            max_in_flight=max_in_flight,
            device=device,
            runtime=cast(NcclRuntimeProtocol, runtime),
        )
    if transport_type == lifecycle_pb2.WORKER_TRANSPORT_TRANSFER_ENGINE:
        runtime = None if runtime_registry is None else runtime_registry.runtime_for(transport_type)
        if runtime is None:
            raise ValueError("Transfer Engine Worker transport requires a process-level runtime")
        return _create_transfer_engine_transport(
            endpoint,
            endpoint_config,
            max_in_flight=max_in_flight,
            device=device,
            runtime=runtime,
            worker_start_id=worker_start_id,
        )
    raise ValueError(f"unsupported Worker transport type: {transport_type}")
