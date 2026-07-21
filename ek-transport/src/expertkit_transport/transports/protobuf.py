"""Protobuf fields shared by gRPC Tensor and SHM notification messages."""

from __future__ import annotations

import torch
from expertkit_proto.ek.worker.v2 import common_pb2, computation_pb2

from expertkit_transport.errors import (
    TransportError,
    TransportErrorCode,
    TransportProtocolError,
)
from expertkit_transport.transports.base import WorkerEndpointConfig

MAX_DIAGNOSTIC_BYTES = 1024

_DTYPE_TO_PROTO = {
    torch.float16: common_pb2.ACTIVATION_DTYPE_FP16,
    torch.bfloat16: common_pb2.ACTIVATION_DTYPE_BF16,
    torch.float32: common_pb2.ACTIVATION_DTYPE_FP32,
}
_PROTO_TO_DTYPE = {wire: dtype for dtype, wire in _DTYPE_TO_PROTO.items()}
_TRANSPORT_TO_PROTO_ERROR = {
    TransportErrorCode.BUSY: computation_pb2.COMPUTE_ERROR_BUSY,
    TransportErrorCode.DRAINING: computation_pb2.COMPUTE_ERROR_DRAINING,
    TransportErrorCode.STALE_TOPOLOGY: computation_pb2.COMPUTE_ERROR_STALE_TOPOLOGY,
    TransportErrorCode.EXPERT_NOT_READY: computation_pb2.COMPUTE_ERROR_EXPERT_NOT_READY,
    TransportErrorCode.INVALID_REQUEST: computation_pb2.COMPUTE_ERROR_INVALID_REQUEST,
    TransportErrorCode.UNSUPPORTED: computation_pb2.COMPUTE_ERROR_UNSUPPORTED,
}
_PROTO_TO_TRANSPORT_ERROR = {wire: code for code, wire in _TRANSPORT_TO_PROTO_ERROR.items()}


def activation_dtype_to_protobuf(dtype: torch.dtype) -> int:
    """Return the protocol enum for one supported activation dtype."""

    try:
        return _DTYPE_TO_PROTO[dtype]
    except KeyError as error:
        raise ValueError("dtype must be FP16, BF16, or FP32") from error


def activation_dtype_from_protobuf(value: int) -> torch.dtype | None:
    """Return the Torch dtype for one known protocol enum."""

    return _PROTO_TO_DTYPE.get(value)


def _bounded_diagnostic(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_DIAGNOSTIC_BYTES:
        return value
    return encoded[:MAX_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")


def encode_compute_error(
    error: TransportError,
    config: WorkerEndpointConfig,
) -> computation_pb2.ComputeError:
    """Return one bounded protobuf computation rejection."""

    try:
        code = _TRANSPORT_TO_PROTO_ERROR[error.code]
    except KeyError as cause:
        raise ValueError("this Transport error must use native RPC status") from cause
    if len(error.unavailable_expert_ids) > config.max_batch_tokens * config.top_k:
        raise ValueError("too many unavailable expert IDs for one Worker batch")
    if any(
        expert_id < 0 or expert_id >= config.experts_per_layer
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
    config: WorkerEndpointConfig,
) -> TransportError:
    """Validate and map one protobuf computation rejection."""

    code = _PROTO_TO_TRANSPORT_ERROR.get(wire_error.code)
    if code is None:
        raise TransportProtocolError("response contains an unknown computation error code")
    if len(wire_error.diagnostic.encode("utf-8")) > MAX_DIAGNOSTIC_BYTES:
        raise TransportProtocolError("response diagnostic exceeds its configured bound")
    if len(wire_error.unavailable_expert_ids) > config.max_batch_tokens * config.top_k:
        raise TransportProtocolError("response contains too many unavailable expert IDs")
    if any(
        expert_id >= config.experts_per_layer for expert_id in wire_error.unavailable_expert_ids
    ):
        raise TransportProtocolError("response unavailable expert ID exceeds the configured range")
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
