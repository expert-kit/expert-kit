"""Tests for v2 gRPC tensor and error encoding."""

import gc

import pytest
import torch

from expertkit_transport._proto.ek.worker.v2 import computation_pb2
from expertkit_transport.adapters.grpc import (
    GrpcBatchSpec,
    GrpcProtocolError,
    decode_request,
    decode_response,
    encode_error_response,
    encode_request,
    encode_success_response,
)
from expertkit_transport.contracts import (
    TransportError,
    TransportErrorCode,
    WorkerBatch,
)


def spec(dtype: torch.dtype = torch.float16) -> GrpcBatchSpec:
    return GrpcBatchSpec(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=dtype,
    )


def worker_batch(dtype: torch.dtype = torch.float16) -> WorkerBatch:
    hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=dtype,
    )
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=hidden_states,
        token_indices=torch.tensor([2, 0], dtype=torch.int64),
        expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0.0], [0.5, 0.5]], dtype=torch.float32),
        distinct_expert_ids=(0, 1, 3),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_request_round_trip_compacts_rows_and_preserves_raw_dtypes(dtype: torch.dtype) -> None:
    source = worker_batch(dtype)

    decoded = decode_request(encode_request(source, spec(dtype)), spec(dtype))
    gc.collect()

    assert decoded.token_indices is None
    assert decoded.hidden_states.dtype == dtype
    torch.testing.assert_close(decoded.hidden_states, source.hidden_states[[2, 0]])
    assert decoded.expert_ids.dtype == torch.int32
    assert decoded.expert_ids.tolist() == [[1, -1], [0, 3]]
    assert decoded.routing_weights.dtype == torch.float32
    torch.testing.assert_close(decoded.routing_weights, source.routing_weights)
    assert decoded.distinct_expert_ids == (0, 1, 3)


def test_request_fields_contain_little_endian_raw_tensors() -> None:
    payload = encode_request(worker_batch(torch.float32), spec(torch.float32))
    request = computation_pb2.ExecuteRequest.FromString(payload)

    assert request.token_count == 2
    assert request.hidden_dim == 3
    assert request.top_k == 2
    assert request.hidden_states == bytes.fromhex(
        "0000e04000000041000010410000803f0000004000004040"
    )
    assert request.expert_ids == bytes.fromhex("01000000ffffffff0000000003000000")


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("instance_id", 8, "unknown model instance"),
        ("layer_id", 4, "layer ID"),
        ("token_count", 0, "token_count"),
        ("hidden_dim", 4, "hidden_dim"),
        ("top_k", 1, "top_k"),
        ("dtype", 0, "dtype"),
    ],
)
def test_decode_rejects_mismatched_metadata(field: str, value: int, diagnostic: str) -> None:
    request = computation_pb2.ExecuteRequest.FromString(encode_request(worker_batch(), spec()))
    setattr(request, field, value)

    with pytest.raises(GrpcProtocolError, match=diagnostic):
        decode_request(request.SerializeToString(), spec())


@pytest.mark.parametrize("field", ["hidden_states", "expert_ids", "routing_weights"])
def test_decode_rejects_each_malformed_tensor_length(field: str) -> None:
    request = computation_pb2.ExecuteRequest.FromString(encode_request(worker_batch(), spec()))
    setattr(request, field, getattr(request, field)[:-1])

    with pytest.raises(GrpcProtocolError, match="encoded length"):
        decode_request(request.SerializeToString(), spec())


def test_decode_rejects_malformed_protobuf() -> None:
    with pytest.raises(GrpcProtocolError, match="valid protobuf"):
        decode_request(b"\xff", spec())


@pytest.mark.parametrize(
    ("expert_ids", "routing_weights", "diagnostic"),
    [
        (
            torch.tensor([[-2, 0]], dtype=torch.int32),
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            "below -1",
        ),
        (
            torch.tensor([[8, 0]], dtype=torch.int32),
            torch.tensor([[1.0, 1.0]], dtype=torch.float32),
            "expert range",
        ),
        (
            torch.tensor([[-1, 0]], dtype=torch.int32),
            torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            "zero routing weight",
        ),
    ],
)
def test_decode_rejects_invalid_routing_values(
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    diagnostic: str,
) -> None:
    source = WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.ones((1, 3), dtype=torch.float16),
        token_indices=None,
        expert_ids=expert_ids,
        routing_weights=routing_weights,
        distinct_expert_ids=(),
    )
    request = computation_pb2.ExecuteRequest.FromString(encode_request(worker_batch(), spec()))
    request.token_count = 1
    request.hidden_states = source.hidden_states.view(torch.uint8).numpy().tobytes()
    request.expert_ids = expert_ids.view(torch.uint8).numpy().tobytes()
    request.routing_weights = routing_weights.view(torch.uint8).numpy().tobytes()

    with pytest.raises(GrpcProtocolError, match=diagnostic):
        decode_request(request.SerializeToString(), spec())


def test_encode_rejects_distinct_expert_list_drift() -> None:
    source = worker_batch()
    drifted = WorkerBatch(
        instance_id=source.instance_id,
        layer_id=source.layer_id,
        topology_version=source.topology_version,
        hidden_states=source.hidden_states,
        token_indices=source.token_indices,
        expert_ids=source.expert_ids,
        routing_weights=source.routing_weights,
        distinct_expert_ids=(0, 1),
    )

    with pytest.raises(GrpcProtocolError, match="distinct_expert_ids"):
        encode_request(drifted, spec())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_success_response_round_trip_uses_request_shape(dtype: torch.dtype) -> None:
    partial = torch.tensor([[1.25, 2.5, 3.75], [4.0, 5.0, 6.0]], dtype=dtype)

    decoded = decode_response(
        encode_success_response(partial, spec(dtype)),
        token_count=2,
        spec=spec(dtype),
    )
    gc.collect()

    assert decoded.dtype == dtype
    torch.testing.assert_close(decoded, partial)


def test_success_response_rejects_wrong_encoded_length() -> None:
    response = computation_pb2.ExecuteResponse(partial_output=b"too short")

    with pytest.raises(GrpcProtocolError, match="encoded length"):
        decode_response(response.SerializeToString(), token_count=2, spec=spec())


def test_structured_error_round_trip_preserves_recovery_fields() -> None:
    draining = TransportError(
        TransportErrorCode.DRAINING,
        retryable=True,
        observed_topology_version=11,
        min_topology_version=12,
        unavailable_expert_ids=(1, 3),
        diagnostic="route is draining",
    )

    with pytest.raises(TransportError) as caught:
        decode_response(
            encode_error_response(draining, spec()),
            token_count=2,
            spec=spec(),
        )

    assert caught.value.code is TransportErrorCode.DRAINING
    assert caught.value.retryable is True
    assert caught.value.observed_topology_version == 11
    assert caught.value.min_topology_version == 12
    assert caught.value.unavailable_expert_ids == (1, 3)
    assert caught.value.diagnostic == "route is draining"


def test_error_diagnostic_is_bounded_on_encode() -> None:
    error = TransportError(
        TransportErrorCode.BUSY,
        retryable=True,
        diagnostic="界" * 1024,
    )
    payload = encode_error_response(error, spec())
    response = computation_pb2.ExecuteResponse.FromString(payload)

    assert len(response.error.diagnostic.encode("utf-8")) <= 1024


def test_unknown_or_missing_response_branch_is_protocol_error() -> None:
    unknown_error = computation_pb2.ExecuteResponse(
        error=computation_pb2.ComputeError(code=999, retryable=True)
    )

    with pytest.raises(GrpcProtocolError, match="unknown computation error"):
        decode_response(unknown_error.SerializeToString(), token_count=1, spec=spec())
    with pytest.raises(GrpcProtocolError, match="does not contain a result"):
        decode_response(b"", token_count=1, spec=spec())


def test_non_computation_error_cannot_use_structured_response() -> None:
    error = TransportError(TransportErrorCode.UNAVAILABLE, retryable=True)

    with pytest.raises(ValueError, match="native gRPC status"):
        encode_error_response(error, spec())


@pytest.mark.parametrize("unavailable", [(8,), tuple(range(9))])
def test_error_response_rejects_unbounded_recovery_ids(
    unavailable: tuple[int, ...],
) -> None:
    error = TransportError(
        TransportErrorCode.EXPERT_NOT_READY,
        retryable=True,
        unavailable_expert_ids=unavailable,
    )

    with pytest.raises(ValueError, match="unavailable expert"):
        encode_error_response(error, spec())
