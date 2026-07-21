"""Encode and validate v2 gRPC computation messages."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from expertkit_proto.ek.worker.v2 import computation_pb2
from google.protobuf.message import DecodeError

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.grpc.spec import (
    MAX_DIAGNOSTIC_BYTES,
    GrpcBatchSpec,
    calculate_message_limits,
)

_TRANSPORT_TO_PROTO_ERROR = {
    TransportErrorCode.BUSY: computation_pb2.COMPUTE_ERROR_BUSY,
    TransportErrorCode.DRAINING: computation_pb2.COMPUTE_ERROR_DRAINING,
    TransportErrorCode.STALE_TOPOLOGY: computation_pb2.COMPUTE_ERROR_STALE_TOPOLOGY,
    TransportErrorCode.EXPERT_NOT_READY: computation_pb2.COMPUTE_ERROR_EXPERT_NOT_READY,
    TransportErrorCode.INVALID_REQUEST: computation_pb2.COMPUTE_ERROR_INVALID_REQUEST,
    TransportErrorCode.UNSUPPORTED: computation_pb2.COMPUTE_ERROR_UNSUPPORTED,
}
_PROTO_TO_TRANSPORT_ERROR = {wire: code for code, wire in _TRANSPORT_TO_PROTO_ERROR.items()}


class GrpcProtocolError(ValueError):
    """Report malformed or inconsistent gRPC computation data."""


@dataclass(frozen=True, slots=True)
class DecodedRequest:
    """Hold one decoded batch and the bytes backing its Tensor views."""

    batch: WorkerBatch
    retained_tensor_bytes: int


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("the gRPC tensor codec requires a little-endian Host")


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu":
        raise ValueError("gRPC serialization requires a CPU Tensor")
    if not tensor.is_contiguous():
        raise ValueError("gRPC serialization requires a contiguous Tensor")
    return tensor.detach().view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(raw: bytes, dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        flat = torch.frombuffer(raw, dtype=dtype)
    return flat.reshape(shape)


def validate_received_routing(
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    experts_per_layer: int,
) -> tuple[int, ...]:
    """Validate untrusted Host routing values and return distinct expert IDs."""

    expert_values = expert_ids.numpy().reshape(-1)
    routing_values = routing_weights.numpy().reshape(-1)
    if int(expert_values.min()) < -1:
        raise GrpcProtocolError("expert IDs below -1 are invalid")
    if int(expert_values.max()) >= experts_per_layer:
        raise GrpcProtocolError("expert ID exceeds the configured expert range")
    invalid = expert_values == -1
    if np.any(routing_values[invalid] != 0):
        raise GrpcProtocolError("an invalid expert position must have zero routing weight")
    valid = expert_values[~invalid]
    if valid.size == 0:
        return ()
    seen = np.zeros(experts_per_layer, dtype=np.bool_)
    seen[valid] = True
    return tuple(np.flatnonzero(seen).tolist())


def _validate_batch_against_spec(batch: WorkerBatch, spec: GrpcBatchSpec) -> None:
    if batch.instance_id != spec.instance_id:
        raise GrpcProtocolError("Worker batch instance ID does not match the gRPC endpoint")
    if batch.layer_id >= spec.num_layers:
        raise GrpcProtocolError("Worker batch layer ID exceeds the configured layer range")
    if batch.token_count > spec.max_batch_tokens:
        raise GrpcProtocolError("Worker batch exceeds max_batch_tokens")
    if batch.hidden_dim != spec.hidden_dim:
        raise GrpcProtocolError("Worker batch hidden dimension does not match the endpoint")
    if batch.top_k != spec.top_k:
        raise GrpcProtocolError("Worker batch top-k does not match the endpoint")
    if batch.hidden_states.dtype != spec.dtype:
        raise GrpcProtocolError("Worker batch activation dtype does not match the endpoint")


def _serialize_host_request(
    batch: WorkerBatch,
    spec: GrpcBatchSpec,
    host_hidden_states: torch.Tensor,
    host_expert_ids: torch.Tensor,
    host_routing_weights: torch.Tensor,
) -> bytes:
    request = computation_pb2.ExecuteRequest(
        instance_id=batch.instance_id,
        layer_id=batch.layer_id,
        topology_version=batch.topology_version,
        token_count=batch.token_count,
        hidden_dim=batch.hidden_dim,
        top_k=batch.top_k,
        dtype=spec.protobuf_dtype,
        hidden_states=_raw_bytes(host_hidden_states),
        expert_ids=_raw_bytes(host_expert_ids),
        routing_weights=_raw_bytes(host_routing_weights),
    )
    payload = request.SerializeToString()
    if len(payload) > calculate_message_limits(spec).request_bytes:
        raise RuntimeError("encoded request exceeds its calculated gRPC limit")
    return payload


def encode_request(batch: WorkerBatch, spec: GrpcBatchSpec) -> bytes:
    """Compact one Worker batch and return its serialized v2 request."""

    _require_little_endian()
    _validate_batch_against_spec(batch, spec)
    try:
        if batch.token_indices is None:
            hidden_states = batch.hidden_states
        else:
            hidden_states = torch.index_select(
                batch.hidden_states,
                0,
                batch.token_indices,
            )
    except (IndexError, RuntimeError) as error:
        raise GrpcProtocolError("Worker batch token indices are invalid") from error

    host_hidden = hidden_states.detach().to(device="cpu").contiguous()
    host_expert_ids = batch.expert_ids.detach().to(device="cpu").contiguous()
    host_routing_weights = batch.routing_weights.detach().to(device="cpu").contiguous()
    return _serialize_host_request(
        batch,
        spec,
        host_hidden,
        host_expert_ids,
        host_routing_weights,
    )


def decode_request(payload: bytes, spec: GrpcBatchSpec) -> WorkerBatch:
    """Validate serialized v2 input before constructing Host Tensor views."""

    return decode_request_with_size(payload, spec).batch


def decode_request_with_size(payload: bytes, spec: GrpcBatchSpec) -> DecodedRequest:
    """Decode input and report the bytes retained by its Tensor views."""

    _require_little_endian()
    if len(payload) > calculate_message_limits(spec).request_bytes:
        raise GrpcProtocolError("request exceeds the configured gRPC message limit")
    request = computation_pb2.ExecuteRequest()
    try:
        request.ParseFromString(payload)
    except DecodeError as error:
        raise GrpcProtocolError("request is not valid protobuf") from error

    if request.instance_id != spec.instance_id:
        raise GrpcProtocolError("unknown model instance")
    if request.layer_id >= spec.num_layers:
        raise GrpcProtocolError("layer ID exceeds the configured layer range")
    if not 0 < request.token_count <= spec.max_batch_tokens:
        raise GrpcProtocolError("token_count must be positive and within max_batch_tokens")
    if request.hidden_dim != spec.hidden_dim:
        raise GrpcProtocolError("hidden_dim does not match the configured model")
    if request.top_k != spec.top_k:
        raise GrpcProtocolError("top_k does not match the configured model")
    wire_dtype = GrpcBatchSpec.dtype_from_protobuf(request.dtype)
    if wire_dtype is None or wire_dtype != spec.dtype:
        raise GrpcProtocolError("dtype is unknown or does not match the configured model")

    token_count = request.token_count
    hidden_bytes = token_count * spec.hidden_dim * spec.activation_element_bytes
    routing_bytes = token_count * spec.top_k * 4
    if len(request.hidden_states) != hidden_bytes:
        raise GrpcProtocolError("hidden_states has an inconsistent encoded length")
    if len(request.expert_ids) != routing_bytes:
        raise GrpcProtocolError("expert_ids has an inconsistent encoded length")
    if len(request.routing_weights) != routing_bytes:
        raise GrpcProtocolError("routing_weights has an inconsistent encoded length")

    hidden_states = _tensor_from_bytes(
        request.hidden_states,
        spec.dtype,
        (token_count, spec.hidden_dim),
    )
    expert_ids = _tensor_from_bytes(
        request.expert_ids,
        torch.int32,
        (token_count, spec.top_k),
    )
    routing_weights = _tensor_from_bytes(
        request.routing_weights,
        torch.float32,
        (token_count, spec.top_k),
    )
    distinct = validate_received_routing(
        expert_ids,
        routing_weights,
        spec.experts_per_layer,
    )
    return DecodedRequest(
        batch=WorkerBatch(
            instance_id=request.instance_id,
            layer_id=request.layer_id,
            topology_version=request.topology_version,
            hidden_states=hidden_states,
            token_indices=None,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            distinct_expert_ids=distinct,
        ),
        retained_tensor_bytes=hidden_bytes + routing_bytes + routing_bytes,
    )


def _bounded_diagnostic(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_DIAGNOSTIC_BYTES:
        return value
    return encoded[:MAX_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")


def encode_compute_error(
    error: TransportError,
    spec: GrpcBatchSpec,
) -> computation_pb2.ComputeError:
    """Return one bounded protobuf computation rejection."""

    try:
        code = _TRANSPORT_TO_PROTO_ERROR[error.code]
    except KeyError as cause:
        raise ValueError("this Transport error must use native gRPC status") from cause
    if len(error.unavailable_expert_ids) > spec.max_batch_tokens * spec.top_k:
        raise ValueError("too many unavailable expert IDs for one Worker batch")
    if any(
        expert_id < 0 or expert_id >= spec.experts_per_layer
        for expert_id in error.unavailable_expert_ids
    ):
        raise ValueError("unavailable expert ID exceeds the configured expert range")

    fields: dict[str, object] = {
        "code": code,
        "retryable": error.retryable,
        "unavailable_expert_ids": error.unavailable_expert_ids,
        "diagnostic": _bounded_diagnostic(error.diagnostic),
    }
    if error.observed_topology_version is not None:
        fields["observed_topology_version"] = error.observed_topology_version
    if error.min_topology_version is not None:
        fields["min_topology_version"] = error.min_topology_version
    return computation_pb2.ComputeError(**fields)


def decode_compute_error(
    wire_error: computation_pb2.ComputeError,
    spec: GrpcBatchSpec,
) -> TransportError:
    """Validate and map one protobuf computation rejection."""

    code = _PROTO_TO_TRANSPORT_ERROR.get(wire_error.code)
    if code is None:
        raise GrpcProtocolError("response contains an unknown computation error code")
    if len(wire_error.diagnostic.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
        raise GrpcProtocolError("response diagnostic exceeds its configured bound")
    if len(wire_error.unavailable_expert_ids) > spec.max_batch_tokens * spec.top_k:
        raise GrpcProtocolError("response contains too many unavailable expert IDs")
    if any(expert_id >= spec.experts_per_layer for expert_id in wire_error.unavailable_expert_ids):
        raise GrpcProtocolError("response unavailable expert ID exceeds the configured range")
    return TransportError(
        code,
        retryable=wire_error.retryable,
        observed_topology_version=(
            wire_error.observed_topology_version
            if wire_error.HasField("observed_topology_version")
            else None
        ),
        min_topology_version=(
            wire_error.min_topology_version if wire_error.HasField("min_topology_version") else None
        ),
        unavailable_expert_ids=tuple(wire_error.unavailable_expert_ids),
        diagnostic=wire_error.diagnostic,
    )


def encode_error_response(error: TransportError, spec: GrpcBatchSpec) -> bytes:
    """Return a bounded structured computation rejection."""

    response = computation_pb2.ExecuteResponse(
        error=encode_compute_error(error, spec),
    )
    payload = response.SerializeToString()
    if len(payload) > calculate_message_limits(spec).response_bytes:
        raise RuntimeError("encoded error exceeds its calculated gRPC limit")
    return payload


def encode_success_response(partial_output: torch.Tensor, spec: GrpcBatchSpec) -> bytes:
    """Return a serialized activation-dtype partial output."""

    _require_little_endian()
    if partial_output.ndim != 2 or partial_output.shape[1] != spec.hidden_dim:
        raise ValueError("partial output must have shape [token_count, hidden_dim]")
    if not 0 < partial_output.shape[0] <= spec.max_batch_tokens:
        raise ValueError("partial output token count exceeds the configured limit")
    if partial_output.dtype != spec.dtype:
        raise ValueError("partial output dtype does not match the configured model")
    response = computation_pb2.ExecuteResponse(partial_output=_raw_bytes(partial_output))
    payload = response.SerializeToString()
    if len(payload) > calculate_message_limits(spec).response_bytes:
        raise RuntimeError("encoded response exceeds its calculated gRPC limit")
    return payload


def decode_response(payload: bytes, token_count: int, spec: GrpcBatchSpec) -> torch.Tensor:
    """Return a validated CPU partial view or raise its structured rejection."""

    _require_little_endian()
    if not 0 < token_count <= spec.max_batch_tokens:
        raise ValueError("token_count must be positive and within max_batch_tokens")
    if len(payload) > calculate_message_limits(spec).response_bytes:
        raise GrpcProtocolError("response exceeds the configured gRPC message limit")
    response = computation_pb2.ExecuteResponse()
    try:
        response.ParseFromString(payload)
    except DecodeError as error:
        raise GrpcProtocolError("response is not valid protobuf") from error

    result = response.WhichOneof("result")
    if result == "error":
        raise decode_compute_error(response.error, spec)
    if result != "partial_output":
        raise GrpcProtocolError("response does not contain a result")

    expected_bytes = token_count * spec.hidden_dim * spec.activation_element_bytes
    if len(response.partial_output) != expected_bytes:
        raise GrpcProtocolError("partial_output has an inconsistent encoded length")
    return _tensor_from_bytes(
        response.partial_output,
        spec.dtype,
        (token_count, spec.hidden_dim),
    )
