"""Validated gRPC computation shape and finite message limits."""

from __future__ import annotations

from dataclasses import dataclass

from expertkit_transport.transports.base import WorkerEndpointConfig
from expertkit_transport.transports.protobuf import (
    MAX_DIAGNOSTIC_BYTES,
    activation_dtype_to_protobuf,
)

_GRPC_MAX_MESSAGE_BYTES = (1 << 31) - 1
_PROTOBUF_SAFETY_MARGIN_BYTES = 256


def _checked_add(*values: int) -> int:
    total = sum(values)
    if total > _GRPC_MAX_MESSAGE_BYTES:
        raise ValueError("configured gRPC message size exceeds the signed 32-bit limit")
    return total


def _checked_multiply(*values: int) -> int:
    result = 1
    for value in values:
        result *= value
        if result > _GRPC_MAX_MESSAGE_BYTES:
            raise ValueError("configured gRPC tensor size exceeds the signed 32-bit limit")
    return result


def _varint_size(value: int) -> int:
    if value < 0:
        raise ValueError("varint values must be nonnegative")
    return max(1, (value.bit_length() + 6) // 7)


def _tag_size(field_number: int, wire_type: int) -> int:
    return _varint_size((field_number << 3) | wire_type)


def _bytes_field_size(field_number: int, payload_bytes: int) -> int:
    return _checked_add(_tag_size(field_number, 2), _varint_size(payload_bytes), payload_bytes)


@dataclass(frozen=True, slots=True)
class GrpcMessageLimits:
    """Hold finite encoded-message limits for both gRPC directions."""

    request_bytes: int
    response_bytes: int
    retained_request_tensor_bytes: int

    @property
    def client_options(self) -> tuple[tuple[str, int], ...]:
        """Return client send and receive limits for `grpc.aio.insecure_channel`."""

        return (
            ("grpc.max_send_message_length", self.request_bytes),
            ("grpc.max_receive_message_length", self.response_bytes),
        )

    @property
    def server_options(self) -> tuple[tuple[str, int], ...]:
        """Return server receive and send limits for `grpc.aio.server`."""

        return (
            ("grpc.max_receive_message_length", self.request_bytes),
            ("grpc.max_send_message_length", self.response_bytes),
        )


def calculate_message_limits(spec: WorkerEndpointConfig) -> GrpcMessageLimits:
    """Calculate finite protobuf limits without allocating maximum-size payloads."""

    token_count = spec.max_batch_tokens
    hidden_bytes = _checked_multiply(
        token_count,
        spec.hidden_dim,
        spec.activation_element_bytes,
    )
    expert_bytes = _checked_multiply(token_count, spec.top_k, 4)
    routing_bytes = _checked_multiply(token_count, spec.top_k, 4)
    retained = _checked_add(hidden_bytes, expert_bytes, routing_bytes)

    scalar_fields = (
        _tag_size(1, 0) + 10,
        _tag_size(2, 0) + 5,
        _tag_size(3, 0) + 10,
        _tag_size(4, 0) + _varint_size(token_count),
        _tag_size(5, 0) + _varint_size(spec.hidden_dim),
        _tag_size(6, 0) + _varint_size(spec.top_k),
        _tag_size(7, 0) + _varint_size(activation_dtype_to_protobuf(spec.dtype)),
    )
    request_bytes = _checked_add(
        *scalar_fields,
        _bytes_field_size(8, hidden_bytes),
        _bytes_field_size(9, expert_bytes),
        _bytes_field_size(10, routing_bytes),
        _PROTOBUF_SAFETY_MARGIN_BYTES,
    )

    success_bytes = _bytes_field_size(1, hidden_bytes)
    unavailable_count = _checked_multiply(token_count, spec.top_k)
    packed_unavailable_bytes = _checked_multiply(unavailable_count, 5)
    error_payload_bytes = _checked_add(
        _tag_size(1, 0) + 1,
        _tag_size(2, 0) + 1,
        _tag_size(3, 0) + 10,
        _tag_size(4, 0) + 10,
        _bytes_field_size(5, packed_unavailable_bytes),
        _bytes_field_size(6, MAX_DIAGNOSTIC_BYTES),
    )
    error_bytes = _bytes_field_size(2, error_payload_bytes)
    response_bytes = _checked_add(
        max(success_bytes, error_bytes),
        _PROTOBUF_SAFETY_MARGIN_BYTES,
    )
    return GrpcMessageLimits(
        request_bytes=request_bytes,
        response_bytes=response_bytes,
        retained_request_tensor_bytes=retained,
    )
