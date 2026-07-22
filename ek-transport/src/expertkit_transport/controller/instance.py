"""Resolve the model instance managed by one Controller."""

from __future__ import annotations

import math
from dataclasses import dataclass

import grpc
from expertkit_proto.ek.control.v2 import lifecycle_pb2, lifecycle_pb2_grpc

from expertkit_transport.errors import TransportError, TransportErrorCode

_CONTROL_MESSAGE_BYTES = 1024 * 1024
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class ResolvedDefaultInstance:
    """Describe the Controller-selected runtime model instance."""

    instance_id: int
    model_name: str
    instance_name: str


async def resolve_default_instance(
    controller_endpoint: str,
    *,
    requested_instance_id: int | None,
    timeout_seconds: float,
) -> ResolvedDefaultInstance:
    """Resolve and validate the Controller's single configured instance.

    Args:
        requested_instance_id: Optional compatibility ID. A supplied value must
            equal the Controller default.
        timeout_seconds: Complete deadline for channel readiness and resolution.

    Raises:
        ValueError: Local arguments are malformed.
        TransportError: The Controller rejects or cannot complete resolution.
    """

    if not isinstance(controller_endpoint, str) or not controller_endpoint.strip():
        raise ValueError("controller_endpoint must not be empty")
    if requested_instance_id is not None and (
        isinstance(requested_instance_id, bool)
        or not isinstance(requested_instance_id, int)
        or not 0 < requested_instance_id <= _UINT64_MAX
    ):
        raise ValueError("requested_instance_id must be a positive uint64 or None")
    if not isinstance(timeout_seconds, int | float) or isinstance(timeout_seconds, bool):
        raise ValueError("timeout_seconds must be finite and positive")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")

    channel = grpc.aio.insecure_channel(
        controller_endpoint,
        options=(
            ("grpc.max_send_message_length", _CONTROL_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _CONTROL_MESSAGE_BYTES),
        ),
    )
    try:
        stub = lifecycle_pb2_grpc.InstanceServiceStub(channel)
        response = await stub.ResolveDefaultInstance(
            lifecycle_pb2.ResolveDefaultInstanceRequest(
                requested_instance_id=requested_instance_id or 0,
            ),
            timeout=timeout_seconds,
            wait_for_ready=True,
        )
    except grpc.aio.AioRpcError as error:
        raise _resolution_error(error) from error
    finally:
        await channel.close()

    instance_id = int(response.instance_id)
    if instance_id <= 0 or instance_id > _UINT64_MAX:
        raise TransportError(
            TransportErrorCode.PROTOCOL,
            retryable=False,
            diagnostic="Controller returned an invalid default instance ID",
        )
    if not response.model_name.strip() or not response.instance_name.strip():
        raise TransportError(
            TransportErrorCode.PROTOCOL,
            retryable=False,
            diagnostic="Controller returned incomplete default instance metadata",
        )
    if requested_instance_id is not None and requested_instance_id != instance_id:
        raise TransportError(
            TransportErrorCode.PROTOCOL,
            retryable=False,
            diagnostic="Controller accepted an explicit ID but returned another instance",
        )
    return ResolvedDefaultInstance(
        instance_id=instance_id,
        model_name=response.model_name,
        instance_name=response.instance_name,
    )


def _resolution_error(error: grpc.aio.AioRpcError) -> TransportError:
    code = error.code()
    diagnostic = f"default instance resolution failed: {error.details() or code.name}"
    if code is grpc.StatusCode.DEADLINE_EXCEEDED:
        return TransportError(
            TransportErrorCode.DEADLINE_EXCEEDED,
            retryable=False,
            diagnostic=diagnostic,
        )
    if code is grpc.StatusCode.CANCELLED:
        return TransportError(
            TransportErrorCode.CANCELLED,
            retryable=False,
            diagnostic=diagnostic,
        )
    if code in (
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.NOT_FOUND,
        grpc.StatusCode.FAILED_PRECONDITION,
    ):
        return TransportError(
            TransportErrorCode.INVALID_REQUEST,
            retryable=False,
            diagnostic=diagnostic,
        )
    return TransportError(
        TransportErrorCode.UNAVAILABLE,
        retryable=True,
        diagnostic=diagnostic,
    )
