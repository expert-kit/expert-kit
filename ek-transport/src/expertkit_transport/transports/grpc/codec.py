"""Encode and validate v2 gRPC computation messages."""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import torch
from expertkit_proto.ek.worker.v2 import computation_pb2
from google.protobuf.message import DecodeError

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportProtocolError
from expertkit_transport.transports.base import WorkerEndpointConfig
from expertkit_transport.transports.grpc.spec import (
    calculate_message_limits,
)
from expertkit_transport.transports.protobuf import (
    activation_dtype_from_protobuf,
    activation_dtype_to_protobuf,
    decode_compute_error,
    encode_compute_error,
)
from expertkit_transport.transports.validation import (
    validate_received_routing,
    validate_worker_batch,
)


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


def _serialize_host_request(
    batch: WorkerBatch,
    spec: WorkerEndpointConfig,
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
        dtype=activation_dtype_to_protobuf(spec.dtype),
        hidden_states=_raw_bytes(host_hidden_states),
        expert_ids=_raw_bytes(host_expert_ids),
        routing_weights=_raw_bytes(host_routing_weights),
    )
    payload = request.SerializeToString()
    if len(payload) > calculate_message_limits(spec).request_bytes:
        raise RuntimeError("encoded request exceeds its calculated gRPC limit")
    return payload


def encode_request(batch: WorkerBatch, spec: WorkerEndpointConfig) -> bytes:
    """Compact one Worker batch and return its serialized v2 request."""

    _require_little_endian()
    validate_worker_batch(batch, spec)
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
        raise TransportProtocolError("Worker batch token indices are invalid") from error

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


def decode_request(payload: bytes, spec: WorkerEndpointConfig) -> WorkerBatch:
    """Validate serialized v2 input before constructing Host Tensor views."""

    return decode_request_with_size(payload, spec).batch


def decode_request_with_size(payload: bytes, spec: WorkerEndpointConfig) -> DecodedRequest:
    """Decode input and report the bytes retained by its Tensor views."""

    _require_little_endian()
    if len(payload) > calculate_message_limits(spec).request_bytes:
        raise TransportProtocolError("request exceeds the configured gRPC message limit")
    request = computation_pb2.ExecuteRequest()
    try:
        request.ParseFromString(payload)
    except DecodeError as error:
        raise TransportProtocolError("request is not valid protobuf") from error

    if request.instance_id != spec.instance_id:
        raise TransportProtocolError("unknown model instance")
    if request.layer_id >= spec.num_layers:
        raise TransportProtocolError("layer ID exceeds the configured layer range")
    if not 0 < request.token_count <= spec.max_batch_tokens:
        raise TransportProtocolError("token_count must be positive and within max_batch_tokens")
    if request.hidden_dim != spec.hidden_dim:
        raise TransportProtocolError("hidden_dim does not match the configured model")
    if request.top_k != spec.top_k:
        raise TransportProtocolError("top_k does not match the configured model")
    wire_dtype = activation_dtype_from_protobuf(request.dtype)
    if wire_dtype is None or wire_dtype != spec.dtype:
        raise TransportProtocolError("dtype is unknown or does not match the configured model")

    token_count = request.token_count
    hidden_bytes = token_count * spec.hidden_dim * spec.activation_element_bytes
    routing_bytes = token_count * spec.top_k * 4
    if len(request.hidden_states) != hidden_bytes:
        raise TransportProtocolError("hidden_states has an inconsistent encoded length")
    if len(request.expert_ids) != routing_bytes:
        raise TransportProtocolError("expert_ids has an inconsistent encoded length")
    if len(request.routing_weights) != routing_bytes:
        raise TransportProtocolError("routing_weights has an inconsistent encoded length")

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


def encode_error_response(error: TransportError, spec: WorkerEndpointConfig) -> bytes:
    """Return a bounded structured computation rejection."""

    response = computation_pb2.ExecuteResponse(
        error=encode_compute_error(error, spec),
    )
    payload = response.SerializeToString()
    if len(payload) > calculate_message_limits(spec).response_bytes:
        raise RuntimeError("encoded error exceeds its calculated gRPC limit")
    return payload


def encode_success_response(partial_output: torch.Tensor, spec: WorkerEndpointConfig) -> bytes:
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


def decode_response(payload: bytes, token_count: int, spec: WorkerEndpointConfig) -> torch.Tensor:
    """Return a validated CPU partial view or raise its structured rejection."""

    _require_little_endian()
    if not 0 < token_count <= spec.max_batch_tokens:
        raise ValueError("token_count must be positive and within max_batch_tokens")
    if len(payload) > calculate_message_limits(spec).response_bytes:
        raise TransportProtocolError("response exceeds the configured gRPC message limit")
    response = computation_pb2.ExecuteResponse()
    try:
        response.ParseFromString(payload)
    except DecodeError as error:
        raise TransportProtocolError("response is not valid protobuf") from error

    result = response.WhichOneof("result")
    if result == "error":
        raise decode_compute_error(response.error, spec)
    if result != "partial_output":
        raise TransportProtocolError("response does not contain a result")

    expected_bytes = token_count * spec.hidden_dim * spec.activation_element_bytes
    if len(response.partial_output) != expected_bytes:
        raise TransportProtocolError("partial_output has an inconsistent encoded length")
    return _tensor_from_bytes(
        response.partial_output,
        spec.dtype,
        (token_count, spec.hidden_dim),
    )
