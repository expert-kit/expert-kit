"""Protocol tests for the v2 computation request and response."""

import pytest
from google.protobuf.message import DecodeError

from expertkit_transport._proto.ek.worker.v2 import (
    common_pb2,
    computation_pb2,
    computation_pb2_grpc,
)


def test_computation_service_is_unary() -> None:
    method = computation_pb2.DESCRIPTOR.services_by_name["ComputationService"].methods_by_name[
        "Execute"
    ]

    assert method.client_streaming is False
    assert method.server_streaming is False
    assert hasattr(computation_pb2_grpc, "ComputationServiceStub")


def test_execute_request_field_numbers_are_stable() -> None:
    fields = computation_pb2.ExecuteRequest.DESCRIPTOR.fields_by_name

    assert {name: field.number for name, field in fields.items()} == {
        "instance_id": 1,
        "layer_id": 2,
        "topology_version": 3,
        "token_count": 4,
        "hidden_dim": 5,
        "top_k": 6,
        "dtype": 7,
        "hidden_states": 8,
        "expert_ids": 9,
        "routing_weights": 10,
    }
    assert "request_id" not in fields
    assert "attempt_id" not in fields
    assert "model_name" not in fields


def test_execute_request_round_trip_preserves_raw_tensor_fields() -> None:
    request = computation_pb2.ExecuteRequest(
        instance_id=7,
        layer_id=3,
        topology_version=11,
        token_count=2,
        hidden_dim=4,
        top_k=2,
        dtype=common_pb2.ACTIVATION_DTYPE_BF16,
        hidden_states=b"h" * 16,
        expert_ids=b"e" * 16,
        routing_weights=b"r" * 16,
    )

    decoded = computation_pb2.ExecuteRequest.FromString(request.SerializeToString())

    assert decoded == request
    assert decoded.dtype == common_pb2.ACTIVATION_DTYPE_BF16


def test_execute_response_result_is_exclusive() -> None:
    response = computation_pb2.ExecuteResponse(partial_output=b"result")
    assert response.WhichOneof("result") == "partial_output"

    response.error.CopyFrom(
        computation_pb2.ComputeError(
            code=computation_pb2.COMPUTE_ERROR_EXPERT_NOT_READY,
            retryable=True,
            unavailable_expert_ids=[2, 5],
        )
    )

    assert response.WhichOneof("result") == "error"
    assert response.partial_output == b""
    assert list(response.error.unavailable_expert_ids) == [2, 5]


def test_optional_recovery_versions_preserve_presence() -> None:
    error = computation_pb2.ComputeError(code=computation_pb2.COMPUTE_ERROR_DRAINING)
    assert error.HasField("min_topology_version") is False

    error.min_topology_version = 19

    assert error.HasField("min_topology_version") is True
    assert error.min_topology_version == 19


def test_generated_parser_rejects_malformed_protobuf() -> None:
    with pytest.raises(DecodeError):
        computation_pb2.ExecuteRequest.FromString(b"\x42\xff")
