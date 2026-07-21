"""Create Frontend connections for Worker Transport types published by Controller."""

from __future__ import annotations

import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.transports.base import WorkerEndpointConfig, WorkerTransport
from expertkit_transport.transports.grpc.client import GrpcWorkerTransport
from expertkit_transport.transports.shm.client import ShmWorkerTransport


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
    raise ValueError(f"unsupported Worker transport type: {transport_type}")
