"""Tests for bounded Frontend gRPC output and staging buffers."""

import pytest
import torch

from expertkit_transport.buffers import TorchOutputBufferProvider
from expertkit_transport.buffers.base import OutputSpec
from expertkit_transport.transports.grpc import GrpcBatchSpec
from expertkit_transport.transports.grpc.buffers import (
    GrpcOutputBufferProvider,
    GrpcPreparedOutput,
)


def batch_spec() -> GrpcBatchSpec:
    return GrpcBatchSpec(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float16,
    )


def output_spec(**overrides: object) -> OutputSpec:
    values: dict[str, object] = {
        "max_batch_tokens": 4,
        "hidden_dim": 3,
        "dtype": torch.float16,
        "device": "cpu",
    }
    values.update(overrides)
    return OutputSpec(**values)


def test_cpu_output_owns_fixed_request_and_response_staging() -> None:
    provider = GrpcOutputBufferProvider(batch_spec())
    spec = output_spec()

    output = provider.prepare(spec)

    assert isinstance(output, GrpcPreparedOutput)
    assert output.tensor.shape == (4, 3)
    assert output.host_hidden_states.shape == (4, 3)
    assert output.host_expert_ids.shape == (4, 2)
    assert output.host_expert_ids.dtype == torch.int32
    assert output.host_routing_weights.shape == (4, 2)
    assert output.host_routing_weights.dtype == torch.float32
    assert output.host_partial_output.shape == (4, 3)
    assert output.host_partial_output.data_ptr() == output.tensor.data_ptr()
    assert output.request_copy_event is None
    assert output.receive_event is None
    assert output.consume_event is None
    provider.validate(output, spec)
    provider.before_receive(output)
    provider.after_consume(output)
    provider.release(output)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("max_batch_tokens", 3, "max_batch_tokens"),
        ("hidden_dim", 4, "hidden dimension"),
        ("dtype", torch.float32, "dtype"),
    ],
)
def test_provider_rejects_output_spec_drift(
    field: str,
    value: object,
    diagnostic: str,
) -> None:
    provider = GrpcOutputBufferProvider(batch_spec())

    with pytest.raises(ValueError, match=diagnostic):
        provider.prepare(output_spec(**{field: value}))


def test_provider_rejects_another_adapters_output() -> None:
    provider = GrpcOutputBufferProvider(batch_spec())
    spec = output_spec()
    foreign = TorchOutputBufferProvider().prepare(spec)

    with pytest.raises(TypeError, match="gRPC prepared output"):
        provider.validate(foreign, spec)
    TorchOutputBufferProvider().release(foreign)
