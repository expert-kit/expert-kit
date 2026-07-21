"""Validated gRPC computation shape and finite message limits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from expertkit_proto.ek.worker.v2 import common_pb2

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_GRPC_MAX_MESSAGE_BYTES = (1 << 31) - 1
_PROTOBUF_SAFETY_MARGIN_BYTES = 256
MAX_DIAGNOSTIC_BYTES = 1024

_DTYPE_TO_PROTO = {
    torch.float16: common_pb2.ACTIVATION_DTYPE_FP16,
    torch.bfloat16: common_pb2.ACTIVATION_DTYPE_BF16,
    torch.float32: common_pb2.ACTIVATION_DTYPE_FP32,
}
_PROTO_TO_DTYPE = {wire: dtype for dtype, wire in _DTYPE_TO_PROTO.items()}


def _require_unsigned(name: str, value: int, maximum: int, *, positive: bool = False) -> None:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


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
class GrpcBatchSpec:
    """Fix the authoritative model shape accepted by one gRPC endpoint."""

    instance_id: int
    num_layers: int
    experts_per_layer: int
    max_batch_tokens: int
    hidden_dim: int
    top_k: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        _require_unsigned("instance_id", self.instance_id, _UINT64_MAX, positive=True)
        for name in (
            "num_layers",
            "experts_per_layer",
            "max_batch_tokens",
            "hidden_dim",
            "top_k",
        ):
            _require_unsigned(name, getattr(self, name), _UINT32_MAX, positive=True)
        if self.top_k > self.experts_per_layer:
            raise ValueError("top_k must not exceed experts_per_layer")
        if self.dtype not in _DTYPE_TO_PROTO:
            raise ValueError("dtype must be FP16, BF16, or FP32")
        calculate_message_limits(self)

    @property
    def activation_element_bytes(self) -> int:
        """Return the raw-byte width of one activation element."""

        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def protobuf_dtype(self) -> int:
        """Return the v2 wire enum matching the Torch activation dtype."""

        return _DTYPE_TO_PROTO[self.dtype]

    @classmethod
    def dtype_from_protobuf(cls, value: int) -> torch.dtype | None:
        """Return the Torch dtype for a known v2 wire enum."""

        return _PROTO_TO_DTYPE.get(value)


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


def calculate_message_limits(spec: GrpcBatchSpec) -> GrpcMessageLimits:
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
        _tag_size(7, 0) + _varint_size(spec.protobuf_dtype),
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
