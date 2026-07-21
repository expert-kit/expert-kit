"""Tests for finite gRPC message-size calculations."""

import pytest
import torch
from expertkit_proto.ek.worker.v2 import computation_pb2

from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.grpc import calculate_message_limits
from expertkit_transport.transports.protobuf import activation_dtype_to_protobuf


def spec(**overrides: object) -> WorkerEndpointConfig:
    values: dict[str, object] = {
        "instance_id": 7,
        "num_layers": 4,
        "experts_per_layer": 8,
        "max_batch_tokens": 16,
        "hidden_dim": 32,
        "top_k": 2,
        "dtype": torch.float16,
    }
    values.update(overrides)
    return WorkerEndpointConfig(**values)


def test_limits_cover_both_directions_and_raw_request_storage() -> None:
    limits = calculate_message_limits(spec())
    raw_request = 16 * 32 * 2 + 16 * 2 * 4 + 16 * 2 * 4

    assert limits.retained_request_tensor_bytes == raw_request
    assert limits.request_bytes > raw_request
    assert limits.response_bytes > 16 * 32 * 2
    assert limits.client_options == (
        ("grpc.max_send_message_length", limits.request_bytes),
        ("grpc.max_receive_message_length", limits.response_bytes),
    )
    assert limits.server_options == (
        ("grpc.max_receive_message_length", limits.request_bytes),
        ("grpc.max_send_message_length", limits.response_bytes),
    )


def test_activation_dtype_changes_tensor_and_message_bytes() -> None:
    fp16 = calculate_message_limits(spec(dtype=torch.float16))
    fp32 = calculate_message_limits(spec(dtype=torch.float32))

    assert fp32.retained_request_tensor_bytes - fp16.retained_request_tensor_bytes == 16 * 32 * 2
    assert fp32.request_bytes > fp16.request_bytes
    assert fp32.response_bytes > fp16.response_bytes


def test_limits_cover_maximum_encoded_request_success_and_error() -> None:
    batch_spec = spec(max_batch_tokens=4, hidden_dim=3, top_k=2)
    limits = calculate_message_limits(batch_spec)
    hidden_bytes = 4 * 3 * 2
    routing_bytes = 4 * 2 * 4
    request = computation_pb2.ExecuteRequest(
        instance_id=(1 << 64) - 1,
        layer_id=(1 << 32) - 1,
        topology_version=(1 << 64) - 1,
        token_count=4,
        hidden_dim=3,
        top_k=2,
        dtype=activation_dtype_to_protobuf(batch_spec.dtype),
        hidden_states=b"\xff" * hidden_bytes,
        expert_ids=b"\xff" * routing_bytes,
        routing_weights=b"\xff" * routing_bytes,
    )
    success = computation_pb2.ExecuteResponse(partial_output=b"\xff" * hidden_bytes)
    error = computation_pb2.ExecuteResponse(
        error=computation_pb2.ComputeError(
            code=computation_pb2.COMPUTE_ERROR_EXPERT_NOT_READY,
            retryable=True,
            observed_topology_version=(1 << 64) - 1,
            min_topology_version=(1 << 64) - 1,
            unavailable_expert_ids=[(1 << 32) - 1] * 8,
            diagnostic="x" * 1024,
        )
    )

    assert len(request.SerializeToString()) < limits.request_bytes
    assert len(success.SerializeToString()) < limits.response_bytes
    assert len(error.SerializeToString()) < limits.response_bytes


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("instance_id", 0, "instance_id"),
        ("num_layers", 0, "num_layers"),
        ("max_batch_tokens", -1, "max_batch_tokens"),
        ("top_k", 9, "top_k must not exceed"),
        ("dtype", torch.int8, "dtype must be"),
    ],
)
def test_spec_rejects_invalid_model_shape(field: str, value: object, diagnostic: str) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        spec(**{field: value})


def test_spec_rejects_message_size_above_grpc_integer_limit() -> None:
    with pytest.raises(ValueError, match="signed 32-bit limit"):
        calculate_message_limits(spec(max_batch_tokens=(1 << 30), hidden_dim=8))
