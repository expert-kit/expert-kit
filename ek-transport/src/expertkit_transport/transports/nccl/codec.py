"""Bounded JSON control protocol for NCCL admission and completion."""

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
from expertkit_transport.transports.nccl.runtime import NcclRuntimeProtocol

NCCL_CONTROL_MESSAGE_BYTES = 4096
_PROTOCOL_VERSION = 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class NcclHello:
    rank: int
    world_size: int
    group_name: str
    rendezvous_endpoint: str


@dataclass(frozen=True, slots=True)
class NcclExecute:
    sequence: int
    client_rank: int
    layer_id: int
    topology_version: int
    token_count: int
    timeout_micros: int
    distinct_expert_ids: tuple[int, ...]


def _encode(fields: dict[str, Any]) -> bytes:
    payload = json.dumps(fields, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(payload) > NCCL_CONTROL_MESSAGE_BYTES:
        raise ValueError("NCCL control message exceeds 4096 bytes")
    return payload


def _decode(payload: bytes, expected_kind: str) -> dict[str, Any]:
    if len(payload) > NCCL_CONTROL_MESSAGE_BYTES:
        raise TransportProtocolError("NCCL control message exceeds 4096 bytes")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TransportProtocolError("NCCL control message is not valid JSON") from error
    if not isinstance(value, dict):
        raise TransportProtocolError("NCCL control message must be an object")
    if value.get("version") != _PROTOCOL_VERSION or value.get("kind") != expected_kind:
        raise TransportProtocolError("NCCL control message version or kind is invalid")
    return value


def _integer(fields: dict[str, Any], name: str, *, minimum: int, maximum: int) -> int:
    value = fields.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TransportProtocolError(f"NCCL control field {name} is out of range")
    return value


def _string(fields: dict[str, Any], name: str) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 512:
        raise TransportProtocolError(f"NCCL control field {name} is invalid")
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


def encode_hello_request(
    runtime: NcclRuntimeProtocol,
    spec: WorkerEndpointConfig,
) -> bytes:
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "hello_request",
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "group_name": runtime.group_name,
            **_endpoint_fields(spec),
        }
    )


def decode_hello_request(
    payload: bytes,
    runtime: NcclRuntimeProtocol,
    spec: WorkerEndpointConfig,
) -> NcclHello:
    fields = _decode(payload, "hello_request")
    rank = _integer(fields, "rank", minimum=0, maximum=runtime.world_size - 1)
    world_size = _integer(fields, "world_size", minimum=2, maximum=_UINT32_MAX)
    group_name = _string(fields, "group_name")
    if rank == runtime.rank:
        raise TransportProtocolError("NCCL client and Worker ranks must differ")
    if world_size != runtime.world_size or group_name != runtime.group_name:
        raise TransportProtocolError("NCCL client static world does not match the Worker")
    for name, expected in _endpoint_fields(spec).items():
        if fields.get(name) != expected:
            raise TransportProtocolError(f"NCCL endpoint field {name} does not match")
    return NcclHello(rank, world_size, group_name, runtime.rendezvous_endpoint)


def encode_hello_response(
    runtime: NcclRuntimeProtocol,
    spec: WorkerEndpointConfig,
) -> bytes:
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "hello_response",
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "group_name": runtime.group_name,
            "rendezvous_endpoint": runtime.rendezvous_endpoint,
            **_endpoint_fields(spec),
        }
    )


def decode_hello_response(
    payload: bytes,
    runtime: NcclRuntimeProtocol,
    spec: WorkerEndpointConfig,
) -> NcclHello:
    fields = _decode(payload, "hello_response")
    rank = _integer(fields, "rank", minimum=0, maximum=runtime.world_size - 1)
    world_size = _integer(fields, "world_size", minimum=2, maximum=_UINT32_MAX)
    group_name = _string(fields, "group_name")
    rendezvous_endpoint = _string(fields, "rendezvous_endpoint")
    if rank == runtime.rank:
        raise TransportProtocolError("NCCL client and Worker ranks must differ")
    if (
        world_size != runtime.world_size
        or group_name != runtime.group_name
        or rendezvous_endpoint != runtime.rendezvous_endpoint
    ):
        raise TransportProtocolError("NCCL Worker static world does not match the client")
    for name, expected in _endpoint_fields(spec).items():
        if fields.get(name) != expected:
            raise TransportProtocolError(f"NCCL endpoint field {name} does not match")
    return NcclHello(rank, world_size, group_name, rendezvous_endpoint)


def encode_execute_request(
    request: NcclExecute,
    spec: WorkerEndpointConfig,
) -> bytes:
    if request.sequence <= 0 or request.client_rank < 0:
        raise ValueError("NCCL sequence and client rank are invalid")
    return _encode(
        {
            "version": _PROTOCOL_VERSION,
            "kind": "execute_request",
            "instance_id": spec.instance_id,
            "sequence": request.sequence,
            "client_rank": request.client_rank,
            "layer_id": request.layer_id,
            "topology_version": request.topology_version,
            "token_count": request.token_count,
            "timeout_micros": request.timeout_micros,
            "distinct_expert_ids": list(request.distinct_expert_ids),
        }
    )


def decode_execute_request(
    payload: bytes,
    runtime: NcclRuntimeProtocol,
    spec: WorkerEndpointConfig,
) -> NcclExecute:
    fields = _decode(payload, "execute_request")
    if fields.get("instance_id") != spec.instance_id:
        raise TransportProtocolError("NCCL request instance does not match the Worker")
    sequence = _integer(fields, "sequence", minimum=1, maximum=_UINT64_MAX)
    client_rank = _integer(fields, "client_rank", minimum=0, maximum=runtime.world_size - 1)
    layer_id = _integer(fields, "layer_id", minimum=0, maximum=spec.num_layers - 1)
    topology_version = _integer(
        fields,
        "topology_version",
        minimum=0,
        maximum=_UINT64_MAX,
    )
    token_count = _integer(
        fields,
        "token_count",
        minimum=1,
        maximum=spec.max_batch_tokens,
    )
    timeout_micros = _integer(
        fields,
        "timeout_micros",
        minimum=1,
        maximum=_UINT64_MAX,
    )
    if client_rank == runtime.rank:
        raise TransportProtocolError("NCCL request client rank matches the Worker rank")
    distinct_value = fields.get("distinct_expert_ids")
    if not isinstance(distinct_value, list) or len(distinct_value) > token_count * spec.top_k:
        raise TransportProtocolError("NCCL request distinct expert metadata is invalid")
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
                "NCCL distinct expert IDs must be sorted, unique, and in range"
            )
        distinct.append(value)
        previous = value
    return NcclExecute(
        sequence,
        client_rank,
        layer_id,
        topology_version,
        token_count,
        timeout_micros,
        tuple(distinct),
    )


def encode_admitted(sequence: int) -> bytes:
    return _encode({"version": _PROTOCOL_VERSION, "kind": "admitted", "sequence": sequence})


def decode_admitted(payload: bytes, expected_sequence: int) -> None:
    fields = _decode(payload, "admitted")
    if _integer(fields, "sequence", minimum=1, maximum=_UINT64_MAX) != expected_sequence:
        raise TransportProtocolError("NCCL admission sequence does not match the request")


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
    if _integer(fields, "sequence", minimum=1, maximum=_UINT64_MAX) != expected_sequence:
        raise TransportProtocolError("NCCL terminal sequence does not match the request")
    if fields["kind"] == "success":
        return
    try:
        code = TransportErrorCode(fields.get("code"))
    except ValueError as error:
        raise TransportProtocolError("NCCL response has an unknown error code") from error
    retryable = fields.get("retryable")
    diagnostic = fields.get("diagnostic")
    unavailable = fields.get("unavailable_expert_ids")
    if not isinstance(retryable, bool) or not isinstance(diagnostic, str):
        raise TransportProtocolError("NCCL response error metadata is invalid")
    if not isinstance(unavailable, list) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < spec.experts_per_layer
        for value in unavailable
    ):
        raise TransportProtocolError("NCCL response unavailable experts are invalid")
    observed = fields.get("observed_topology_version")
    minimum = fields.get("min_topology_version")
    for name, value in (("observed", observed), ("minimum", minimum)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _UINT64_MAX
        ):
            raise TransportProtocolError(f"NCCL response {name} topology version is invalid")
    raise TransportError(
        code,
        retryable=retryable,
        observed_topology_version=observed,
        min_topology_version=minimum,
        unavailable_expert_ids=tuple(unavailable),
        diagnostic=diagnostic,
    )


__all__ = [
    "NCCL_CONTROL_MESSAGE_BYTES",
    "NcclExecute",
    "NcclHello",
    "decode_admitted",
    "decode_execute_request",
    "decode_hello_request",
    "decode_hello_response",
    "decode_terminal",
    "encode_admitted",
    "encode_error",
    "encode_execute_request",
    "encode_hello_request",
    "encode_hello_response",
    "encode_success",
]
