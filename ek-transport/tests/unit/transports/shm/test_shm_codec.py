"""Tests for bounded shared-memory control messages."""

import pytest
import torch

from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.grpc import GrpcBatchSpec, GrpcProtocolError
from expertkit_transport.transports.shm.codec import (
    ExecuteSlot,
    decode_execute_request,
    decode_execute_response,
    decode_open_request,
    encode_execute_error,
    encode_execute_request,
    encode_execute_success,
    encode_open_request,
)
from expertkit_transport.transports.shm.memory import SharedMemoryLayout


def spec() -> GrpcBatchSpec:
    return GrpcBatchSpec(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float16,
    )


def layout() -> SharedMemoryLayout:
    return SharedMemoryLayout(2, 4, 3, 2, torch.float16)


def test_open_and_execute_metadata_round_trip() -> None:
    session_id = "a" * 32
    opened = decode_open_request(
        encode_open_request(
            session_id=session_id,
            segment_name=f"expertkit-{'b' * 32}",
            layout=layout(),
            spec=spec(),
        ),
        spec(),
        expected_slot_count=2,
    )
    request = ExecuteSlot(session_id, 1, 3, 2, 9, 4, 5_000_000)

    assert opened.session_id == session_id
    assert opened.layout == layout()
    assert decode_execute_request(encode_execute_request(request), spec()) == request
    decode_execute_response(encode_execute_success(3), 3, spec())


def test_open_rejects_a_layout_that_differs_from_worker_admission() -> None:
    payload = encode_open_request(
        session_id="a" * 32,
        segment_name=f"expertkit-{'b' * 32}",
        layout=layout(),
        spec=spec(),
    )

    with pytest.raises(GrpcProtocolError, match="slot count"):
        decode_open_request(payload, spec(), expected_slot_count=3)


def test_execute_response_preserves_structured_error() -> None:
    payload = encode_execute_error(
        TransportError(
            TransportErrorCode.EXPERT_NOT_READY,
            retryable=True,
            unavailable_expert_ids=(2,),
            diagnostic="not loaded",
        ),
        spec(),
    )

    with pytest.raises(TransportError) as caught:
        decode_execute_response(payload, 1, spec())

    assert caught.value.code is TransportErrorCode.EXPERT_NOT_READY
    assert caught.value.unavailable_expert_ids == (2,)


def test_execute_rejects_a_stale_completion_generation() -> None:
    with pytest.raises(GrpcProtocolError, match="generation"):
        decode_execute_response(encode_execute_success(2), 3, spec())
