"""Tests for fixed gRPC Host staging owned by the connection."""

import pytest
import torch

from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.grpc.client_buffers import GrpcTransferBufferPool


def batch_spec() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float16,
    )


def test_cpu_pool_owns_a_fixed_number_of_request_staging_slots() -> None:
    pool = GrpcTransferBufferPool(batch_spec(), device="cpu", capacity=2)

    assert len(pool.allocated) == 2
    first = pool.take()
    second = pool.take()
    assert first is not second
    assert first.host_hidden_states.shape == (4, 3)
    assert first.host_expert_ids.shape == (4, 2)
    assert first.host_expert_ids.dtype == torch.int32
    assert first.host_routing_weights.shape == (4, 2)
    assert first.host_routing_weights.dtype == torch.float32
    assert first.host_partial_output is None
    assert first.request_copy_event is None
    assert first.receive_event is None
    with pytest.raises(RuntimeError, match="diverged"):
        pool.take()

    pool.put(first)
    assert pool.take() is first
    pool.put(first)
    pool.put(second)
    pool.close()
    pool.close()


def test_pool_rejects_duplicate_or_foreign_returns() -> None:
    pool = GrpcTransferBufferPool(batch_spec(), device="cpu", capacity=1)
    buffers = pool.take()
    pool.put(buffers)

    with pytest.raises(RuntimeError, match="invalid"):
        pool.put(buffers)

    foreign_pool = GrpcTransferBufferPool(batch_spec(), device="cpu", capacity=1)
    foreign = foreign_pool.take()
    with pytest.raises(RuntimeError, match="invalid"):
        pool.put(foreign)
    foreign_pool.put(foreign)
    foreign_pool.close()
    pool.close()
