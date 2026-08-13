"""Bounded JSON control protocol for Transfer Engine sessions and pulls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from expertkit_transport.errors import (
    TransportError,
    TransportErrorCode,
    TransportProtocolError,
)
from expertkit_transport.transports.base import WorkerEndpointConfig
from expertkit_transport.transports.transfer_engine.arena import (
    TransferArenaDescriptor,
    TransferArenaLayout,
)

TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES = 16 * 1024
_PROTOCOL_VERSION = 2
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class TransferOpenSession:
    client_epoch: str
    client_session_id: str
    client_runtime_generation: str
    expected_worker_start_id: str
    backend: str
    arena: TransferArenaDescriptor


@dataclass(frozen=True, slots=True)
class TransferOpenSessionResult:
    worker_start_id: str
    worker_session_id: str
    session_nonce: str
    backend: str
    arena: TransferArenaDescriptor


@dataclass(frozen=True, slots=True)
class TransferExecutePull:
    sequence: int
    client_epoch: str
    session_nonce: str
    expected_worker_start_id: str
    client_slot_index: int
    client_slot_generation: int
    layer_id: int
    topology_version: int
    token_count: int
    timeout_micros: int
    distinct_expert_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransferCloseSession:
    phase: str
    client_epoch: str
    client_session_id: str
    client_runtime_generation: str
    session_nonce: str


def _encode(fields: dict[str, Any]) -> bytes:
    payload = json.dumps(fields, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(payload) > TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES:
        raise ValueError("Transfer Engine control message exceeds 16384 bytes")
    return payload


def _decode(payload: bytes, expected_kind: str) -> dict[str, Any]:
    if len(payload) > TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES:
        raise TransportProtocolError("Transfer Engine control message exceeds 16384 bytes")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransportProtocolError("Transfer Engine control message is not valid JSON") from error
    if not isinstance(value, dict):
        raise TransportProtocolError("Transfer Engine control message must be an object")
    if value.get("version") != _PROTOCOL_VERSION or value.get("kind") != expected_kind:
        raise TransportProtocolError("Transfer Engine control message version or kind is invalid")
    return value


def _integer(fields: dict[str, Any], name: str, minimum: int, maximum: int) -> int:
    value = fields.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TransportProtocolError(f"Transfer Engine field {name} is out of range")
    return value


def _string(fields: dict[str, Any], name: str, *, maximum_bytes: int = 512) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum_bytes:
        raise TransportProtocolError(f"Transfer Engine field {name} is invalid")
    return value


def _endpoint_fields(spec: WorkerEndpointConfig) -> dict[str, int | str]:
    return {
        "instance_id": spec.instance_id,
        "num_layers": spec.num_layers,
        "experts_per_layer": spec.experts_per_layer,
        "max_batch_tokens": spec.max_batch_tokens,
        "hidden_dim": spec.hidden_dim,
        "top_k": spec.top_k,
        "dtype": str(spec.dtype),
    }


def _arena_fields(arena: TransferArenaDescriptor) -> dict[str, int]:
    return {
        "base_address": arena.base_address,
        "slot_count": arena.slot_count,
        "hidden_offset": arena.layout.hidden_offset,
        "expert_ids_offset": arena.layout.expert_ids_offset,
        "routing_weights_offset": arena.layout.routing_weights_offset,
        "output_offset": arena.layout.output_offset,
        "slot_stride": arena.layout.slot_stride,
    }


def _decode_arena(fields: dict[str, Any], spec: WorkerEndpointConfig) -> TransferArenaDescriptor:
    value = fields.get("arena")
    if not isinstance(value, dict):
        raise TransportProtocolError("Transfer Engine arena descriptor is invalid")
    try:
        descriptor = TransferArenaDescriptor(
            base_address=_integer(value, "base_address", 1, _UINT64_MAX),
            slot_count=_integer(value, "slot_count", 1, 4096),
            layout=TransferArenaLayout(
                hidden_offset=_integer(value, "hidden_offset", 0, _UINT64_MAX),
                expert_ids_offset=_integer(value, "expert_ids_offset", 0, _UINT64_MAX),
                routing_weights_offset=_integer(value, "routing_weights_offset", 0, _UINT64_MAX),
                output_offset=_integer(value, "output_offset", 0, _UINT64_MAX),
                slot_stride=_integer(value, "slot_stride", 1, _UINT64_MAX),
            ),
        )
        descriptor.validate(spec)
    except ValueError as error:
        raise TransportProtocolError(str(error)) from error
    return descriptor


def _require_endpoint(fields: dict[str, Any], spec: WorkerEndpointConfig) -> None:
    for name, expected in _endpoint_fields(spec).items():
        if fields.get(name) != expected:
            raise TransportProtocolError(f"Transfer Engine endpoint field {name} does not match")


def encode_open_session_request(
    request: TransferOpenSession,
    spec: WorkerEndpointConfig,
) -> bytes:
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "open_session_request",
            "client_epoch": request.client_epoch,
            "client_session_id": request.client_session_id,
            "client_runtime_generation": request.client_runtime_generation,
            "expected_worker_start_id": request.expected_worker_start_id,
            "backend": request.backend,
            "arena": _arena_fields(request.arena),
            **_endpoint_fields(spec),
        }
    )


def decode_open_session_request(
    payload: bytes,
    spec: WorkerEndpointConfig,
    worker_start_id: str,
    expected_backend: str,
) -> TransferOpenSession:
    fields = _decode(payload, "open_session_request")
    _require_endpoint(fields, spec)
    expected = _string(fields, "expected_worker_start_id")
    if expected != worker_start_id:
        raise TransportProtocolError("Transfer Engine Worker start ID is stale")
    backend = _string(fields, "backend", maximum_bytes=32)
    if backend != expected_backend:
        raise TransportProtocolError("Transfer Engine data backend does not match the Worker")
    return TransferOpenSession(
        client_epoch=_string(fields, "client_epoch"),
        client_session_id=_string(fields, "client_session_id"),
        client_runtime_generation=_string(
            fields,
            "client_runtime_generation",
            maximum_bytes=128,
        ),
        expected_worker_start_id=expected,
        backend=backend,
        arena=_decode_arena(fields, spec),
    )


def encode_open_session_response(
    result: TransferOpenSessionResult,
    spec: WorkerEndpointConfig,
) -> bytes:
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "open_session_response",
            "worker_start_id": result.worker_start_id,
            "worker_session_id": result.worker_session_id,
            "session_nonce": result.session_nonce,
            "backend": result.backend,
            "arena": _arena_fields(result.arena),
            **_endpoint_fields(spec),
        }
    )


def decode_open_session_response(
    payload: bytes,
    spec: WorkerEndpointConfig,
    expected_worker_start_id: str,
    expected_backend: str,
) -> TransferOpenSessionResult:
    fields = _decode(payload, "open_session_response")
    _require_endpoint(fields, spec)
    worker_start_id = _string(fields, "worker_start_id")
    if worker_start_id != expected_worker_start_id:
        raise TransportProtocolError("Transfer Engine Worker start ID changed during open")
    backend = _string(fields, "backend", maximum_bytes=32)
    if backend != expected_backend:
        raise TransportProtocolError("Transfer Engine data backend changed during open")
    return TransferOpenSessionResult(
        worker_start_id=worker_start_id,
        worker_session_id=_string(fields, "worker_session_id"),
        session_nonce=_string(fields, "session_nonce", maximum_bytes=128),
        backend=backend,
        arena=_decode_arena(fields, spec),
    )


def encode_execute_pull_request(
    request: TransferExecutePull,
    spec: WorkerEndpointConfig,
) -> bytes:
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "execute_pull_request",
            "instance_id": spec.instance_id,
            "sequence": request.sequence,
            "client_epoch": request.client_epoch,
            "session_nonce": request.session_nonce,
            "expected_worker_start_id": request.expected_worker_start_id,
            "client_slot_index": request.client_slot_index,
            "client_slot_generation": request.client_slot_generation,
            "layer_id": request.layer_id,
            "topology_version": request.topology_version,
            "token_count": request.token_count,
            "timeout_micros": request.timeout_micros,
            "distinct_expert_ids": list(request.distinct_expert_ids),
        }
    )


def decode_execute_pull_request(
    payload: bytes,
    spec: WorkerEndpointConfig,
    worker_start_id: str,
) -> TransferExecutePull:
    fields = _decode(payload, "execute_pull_request")
    if fields.get("instance_id") != spec.instance_id:
        raise TransportProtocolError("Transfer Engine request instance does not match")
    expected = _string(fields, "expected_worker_start_id")
    if expected != worker_start_id:
        raise TransportProtocolError("Transfer Engine request Worker start ID is stale")
    token_count = _integer(fields, "token_count", 1, spec.max_batch_tokens)
    distinct_value = fields.get("distinct_expert_ids")
    if not isinstance(distinct_value, list) or len(distinct_value) > token_count * spec.top_k:
        raise TransportProtocolError("Transfer Engine distinct expert metadata is invalid")
    distinct: list[int] = []
    previous = -1
    for value in distinct_value:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < spec.experts_per_layer
            or value <= previous
        ):
            raise TransportProtocolError(
                "Transfer Engine distinct expert IDs must be sorted, unique, and in range"
            )
        distinct.append(value)
        previous = value
    return TransferExecutePull(
        sequence=_integer(fields, "sequence", 1, _UINT64_MAX),
        client_epoch=_string(fields, "client_epoch"),
        session_nonce=_string(fields, "session_nonce", maximum_bytes=128),
        expected_worker_start_id=expected,
        client_slot_index=_integer(fields, "client_slot_index", 0, _UINT32_MAX),
        client_slot_generation=_integer(fields, "client_slot_generation", 1, _UINT64_MAX),
        layer_id=_integer(fields, "layer_id", 0, spec.num_layers - 1),
        topology_version=_integer(fields, "topology_version", 0, _UINT64_MAX),
        token_count=token_count,
        timeout_micros=_integer(fields, "timeout_micros", 1, _UINT64_MAX),
        distinct_expert_ids=tuple(distinct),
    )


def encode_admitted(sequence: int) -> bytes:
    return _encode({"version": _PROTOCOL_VERSION, "kind": "admitted", "sequence": sequence})


def decode_admitted(payload: bytes, expected_sequence: int) -> None:
    fields = _decode(payload, "admitted")
    if _integer(fields, "sequence", 1, _UINT64_MAX) != expected_sequence:
        raise TransportProtocolError("Transfer Engine admission sequence does not match")


def encode_success(sequence: int) -> bytes:
    return _encode({"version": _PROTOCOL_VERSION, "kind": "success", "sequence": sequence})


def encode_error(sequence: int, error: TransportError) -> bytes:
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "error",
            "sequence": sequence,
            "code": error.code.value,
            "retryable": error.retryable,
            "observed_topology_version": error.observed_topology_version,
            "min_topology_version": error.min_topology_version,
            "unavailable_expert_ids": list(error.unavailable_expert_ids),
            "diagnostic": error.diagnostic[:1024],
        }
    )


def decode_terminal(
    payload: bytes,
    expected_sequence: int,
    spec: WorkerEndpointConfig,
) -> None:
    try:
        fields = _decode(payload, "success")
    except TransportProtocolError:
        fields = _decode(payload, "error")
    if _integer(fields, "sequence", 1, _UINT64_MAX) != expected_sequence:
        raise TransportProtocolError("Transfer Engine terminal sequence does not match")
    if fields["kind"] == "success":
        return
    try:
        code = TransportErrorCode(fields.get("code"))
    except ValueError as error:
        raise TransportProtocolError(
            "Transfer Engine response has an unknown error code"
        ) from error
    retryable = fields.get("retryable")
    diagnostic = fields.get("diagnostic")
    unavailable = fields.get("unavailable_expert_ids")
    if not isinstance(retryable, bool) or not isinstance(diagnostic, str):
        raise TransportProtocolError("Transfer Engine response error metadata is invalid")
    if not isinstance(unavailable, list) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < spec.experts_per_layer
        for value in unavailable
    ):
        raise TransportProtocolError("Transfer Engine unavailable experts are invalid")
    observed = fields.get("observed_topology_version")
    minimum = fields.get("min_topology_version")
    for name, value in (("observed", observed), ("minimum", minimum)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX
        ):
            raise TransportProtocolError(
                f"Transfer Engine response {name} topology version is invalid"
            )
    raise TransportError(
        code,
        retryable=retryable,
        observed_topology_version=observed,
        min_topology_version=minimum,
        unavailable_expert_ids=tuple(unavailable),
        diagnostic=diagnostic,
    )


def encode_close_session(
    phase: str,
    client_epoch: str,
    client_session_id: str,
    client_runtime_generation: str,
    expected_worker_start_id: str,
    session_nonce: str,
) -> bytes:
    if phase not in {"prepare", "commit"}:
        raise ValueError("Transfer Engine close phase must be prepare or commit")
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "close_session",
            "phase": phase,
            "client_epoch": client_epoch,
            "client_session_id": client_session_id,
            "client_runtime_generation": client_runtime_generation,
            "expected_worker_start_id": expected_worker_start_id,
            "session_nonce": session_nonce,
        }
    )


def decode_close_session(payload: bytes, worker_start_id: str) -> TransferCloseSession:
    fields = _decode(payload, "close_session")
    if _string(fields, "expected_worker_start_id") != worker_start_id:
        raise TransportProtocolError("Transfer Engine close Worker start ID is stale")
    phase = _string(fields, "phase", maximum_bytes=16)
    if phase not in {"prepare", "commit"}:
        raise TransportProtocolError("Transfer Engine close phase is invalid")
    return TransferCloseSession(
        phase=phase,
        client_epoch=_string(fields, "client_epoch"),
        client_session_id=_string(fields, "client_session_id"),
        client_runtime_generation=_string(
            fields,
            "client_runtime_generation",
            maximum_bytes=128,
        ),
        session_nonce=_string(fields, "session_nonce", maximum_bytes=128),
    )


__all__ = [
    "TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES",
    "TransferCloseSession",
    "TransferExecutePull",
    "TransferOpenSession",
    "TransferOpenSessionResult",
    "decode_admitted",
    "decode_close_session",
    "decode_execute_pull_request",
    "decode_open_session_request",
    "decode_open_session_response",
    "decode_terminal",
    "encode_admitted",
    "encode_close_session",
    "encode_error",
    "encode_execute_pull_request",
    "encode_open_session_request",
    "encode_open_session_response",
    "encode_success",
]
