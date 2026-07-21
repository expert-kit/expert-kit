"""Encode and validate shared-memory control messages."""

from __future__ import annotations

from dataclasses import dataclass

from expertkit_proto.ek.worker.v2 import computation_pb2
from google.protobuf.message import DecodeError, Message

from expertkit_transport.errors import TransportError
from expertkit_transport.transports.grpc.codec import (
    GrpcProtocolError,
    decode_compute_error,
    encode_compute_error,
)
from expertkit_transport.transports.grpc.spec import GrpcBatchSpec
from expertkit_transport.transports.shm.memory import SharedMemoryLayout, validate_session_id

SHM_CONTROL_MESSAGE_BYTES = 4096


@dataclass(frozen=True, slots=True)
class OpenSession:
    """Hold validated metadata needed to map one client segment."""

    session_id: str
    segment_name: str
    layout: SharedMemoryLayout


@dataclass(frozen=True, slots=True)
class ExecuteSlot:
    """Hold validated metadata selecting one shared-memory slot."""

    session_id: str
    slot_index: int
    generation: int
    layer_id: int
    topology_version: int
    token_count: int
    timeout_micros: int


def _parse(payload: bytes, message: Message) -> Message:
    if len(payload) > SHM_CONTROL_MESSAGE_BYTES:
        raise GrpcProtocolError("shared-memory control message exceeds 4096 bytes")
    try:
        message.ParseFromString(payload)
    except DecodeError as error:
        raise GrpcProtocolError("shared-memory control message is not valid protobuf") from error
    return message


def encode_open_request(
    *,
    session_id: str,
    segment_name: str,
    layout: SharedMemoryLayout,
    spec: GrpcBatchSpec,
) -> bytes:
    """Encode one session registration request."""

    validate_session_id(session_id)
    request = computation_pb2.OpenSharedMemoryRequest(
        instance_id=spec.instance_id,
        session_id=session_id,
        segment_name=segment_name,
        segment_size=layout.segment_size,
        slot_count=layout.slot_count,
        max_batch_tokens=layout.max_batch_tokens,
        hidden_dim=layout.hidden_dim,
        top_k=layout.top_k,
        dtype=spec.protobuf_dtype,
    )
    return request.SerializeToString()


def decode_open_request(
    payload: bytes,
    spec: GrpcBatchSpec,
    *,
    expected_slot_count: int,
) -> OpenSession:
    """Validate session registration against the fixed Worker endpoint."""

    request = _parse(payload, computation_pb2.OpenSharedMemoryRequest())
    assert isinstance(request, computation_pb2.OpenSharedMemoryRequest)
    try:
        session_id = validate_session_id(request.session_id)
        layout = SharedMemoryLayout(
            slot_count=request.slot_count,
            max_batch_tokens=request.max_batch_tokens,
            hidden_dim=request.hidden_dim,
            top_k=request.top_k,
            dtype=spec.dtype,
        )
    except ValueError as error:
        raise GrpcProtocolError(str(error)) from error
    wire_dtype = GrpcBatchSpec.dtype_from_protobuf(request.dtype)
    if request.instance_id != spec.instance_id:
        raise GrpcProtocolError("shared-memory model instance does not match the Worker")
    if request.slot_count != expected_slot_count:
        raise GrpcProtocolError("shared-memory slot count does not match Worker admission")
    if (
        request.max_batch_tokens != spec.max_batch_tokens
        or request.hidden_dim != spec.hidden_dim
        or request.top_k != spec.top_k
        or wire_dtype != spec.dtype
    ):
        raise GrpcProtocolError("shared-memory Tensor layout does not match the Worker")
    if request.segment_size != layout.segment_size:
        raise GrpcProtocolError("shared-memory segment size does not match its layout")
    return OpenSession(
        session_id=session_id,
        segment_name=request.segment_name,
        layout=layout,
    )


def encode_open_response() -> bytes:
    """Encode an empty successful session registration response."""

    return computation_pb2.OpenSharedMemoryResponse().SerializeToString()


def decode_open_response(payload: bytes) -> None:
    """Validate an empty successful session registration response."""

    _parse(payload, computation_pb2.OpenSharedMemoryResponse())


def encode_execute_request(request: ExecuteSlot) -> bytes:
    """Encode one shared-memory slot notification."""

    validate_session_id(request.session_id)
    return computation_pb2.ExecuteSharedMemoryRequest(
        session_id=request.session_id,
        slot_index=request.slot_index,
        generation=request.generation,
        layer_id=request.layer_id,
        topology_version=request.topology_version,
        token_count=request.token_count,
        timeout_micros=request.timeout_micros,
    ).SerializeToString()


def decode_execute_request(payload: bytes, spec: GrpcBatchSpec) -> ExecuteSlot:
    """Validate one slot notification against fixed model bounds."""

    request = _parse(payload, computation_pb2.ExecuteSharedMemoryRequest())
    assert isinstance(request, computation_pb2.ExecuteSharedMemoryRequest)
    try:
        session_id = validate_session_id(request.session_id)
    except ValueError as error:
        raise GrpcProtocolError(str(error)) from error
    if request.generation == 0:
        raise GrpcProtocolError("shared-memory generation must be positive")
    if request.layer_id >= spec.num_layers:
        raise GrpcProtocolError("shared-memory layer ID exceeds the configured range")
    if not 0 < request.token_count <= spec.max_batch_tokens:
        raise GrpcProtocolError(
            "shared-memory token_count must be positive and within max_batch_tokens"
        )
    if request.timeout_micros == 0:
        raise GrpcProtocolError("shared-memory timeout_micros must be positive")
    return ExecuteSlot(
        session_id=session_id,
        slot_index=request.slot_index,
        generation=request.generation,
        layer_id=request.layer_id,
        topology_version=request.topology_version,
        token_count=request.token_count,
        timeout_micros=request.timeout_micros,
    )


def encode_execute_success(generation: int) -> bytes:
    """Acknowledge that one output slot generation is complete."""

    if generation <= 0:
        raise ValueError("shared-memory generation must be positive")
    return computation_pb2.ExecuteSharedMemoryResponse(
        completed_generation=generation
    ).SerializeToString()


def encode_execute_error(error: TransportError, spec: GrpcBatchSpec) -> bytes:
    """Encode one structured computation rejection."""

    return computation_pb2.ExecuteSharedMemoryResponse(
        error=encode_compute_error(error, spec)
    ).SerializeToString()


def decode_execute_response(payload: bytes, expected_generation: int, spec: GrpcBatchSpec) -> None:
    """Validate completion or raise its structured Worker rejection."""

    response = _parse(payload, computation_pb2.ExecuteSharedMemoryResponse())
    assert isinstance(response, computation_pb2.ExecuteSharedMemoryResponse)
    result = response.WhichOneof("result")
    if result == "error":
        raise decode_compute_error(response.error, spec)
    if result != "completed_generation":
        raise GrpcProtocolError("shared-memory response does not contain a result")
    if response.completed_generation != expected_generation:
        raise GrpcProtocolError("shared-memory response generation does not match the request")


def encode_close_request(session_id: str) -> bytes:
    """Encode one session close request."""

    validate_session_id(session_id)
    return computation_pb2.CloseSharedMemoryRequest(session_id=session_id).SerializeToString()


def decode_close_request(payload: bytes) -> str:
    """Validate and return one session close identifier."""

    request = _parse(payload, computation_pb2.CloseSharedMemoryRequest())
    assert isinstance(request, computation_pb2.CloseSharedMemoryRequest)
    try:
        return validate_session_id(request.session_id)
    except ValueError as error:
        raise GrpcProtocolError(str(error)) from error


def encode_close_response() -> bytes:
    """Encode an empty successful session close response."""

    return computation_pb2.CloseSharedMemoryResponse().SerializeToString()


def decode_close_response(payload: bytes) -> None:
    """Validate an empty successful session close response."""

    _parse(payload, computation_pb2.CloseSharedMemoryResponse())
