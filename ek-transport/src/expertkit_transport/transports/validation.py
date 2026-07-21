"""Validation shared by concrete Transport senders and receivers."""

from __future__ import annotations

import numpy as np
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportProtocolError
from expertkit_transport.transports.base import WorkerEndpointConfig


def validate_received_routing(
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    experts_per_layer: int,
) -> tuple[int, ...]:
    """Validate untrusted Host routing values and return distinct expert IDs."""

    expert_values = expert_ids.numpy().reshape(-1)
    routing_values = routing_weights.numpy().reshape(-1)
    if int(expert_values.min()) < -1:
        raise TransportProtocolError("expert IDs below -1 are invalid")
    if int(expert_values.max()) >= experts_per_layer:
        raise TransportProtocolError("expert ID exceeds the configured expert range")
    invalid = expert_values == -1
    if np.any(routing_values[invalid] != 0):
        raise TransportProtocolError("an invalid expert position must have zero routing weight")
    valid = expert_values[~invalid]
    if valid.size == 0:
        return ()
    seen = np.zeros(experts_per_layer, dtype=np.bool_)
    seen[valid] = True
    return tuple(np.flatnonzero(seen).tolist())


def validate_worker_batch(batch: WorkerBatch, config: WorkerEndpointConfig) -> None:
    """Validate one outbound batch against a Worker's advertised configuration."""

    if batch.instance_id != config.instance_id:
        raise TransportProtocolError("Worker batch instance ID does not match the endpoint")
    if batch.layer_id >= config.num_layers:
        raise TransportProtocolError("Worker batch layer ID exceeds the configured layer range")
    if batch.token_count > config.max_batch_tokens:
        raise TransportProtocolError("Worker batch exceeds max_batch_tokens")
    if batch.hidden_dim != config.hidden_dim:
        raise TransportProtocolError("Worker batch hidden dimension does not match the endpoint")
    if batch.top_k != config.top_k:
        raise TransportProtocolError("Worker batch top-k does not match the endpoint")
    if batch.hidden_states.dtype != config.dtype:
        raise TransportProtocolError("Worker batch activation dtype does not match the endpoint")
