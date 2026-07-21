"""Integration tests for the asynchronous unary gRPC client adapter."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import grpc
import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.grpc import (
    GrpcWorkerTransport,
    decode_request,
    encode_error_response,
    encode_success_response,
)
from expertkit_transport.transports.grpc.spec import calculate_message_limits

RawHandler = Callable[[bytes, grpc.aio.ServicerContext], Awaitable[bytes]]


def batch_spec() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float16,
    )


def worker_batch() -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float16,
        ),
        token_indices=torch.tensor([2, 0], dtype=torch.int64),
        expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0.0], [0.5, 0.5]], dtype=torch.float32),
        distinct_expert_ids=(0, 1, 3),
    )


@asynccontextmanager
async def raw_server(handler: RawHandler):
    limits = calculate_message_limits(batch_spec())
    server = grpc.aio.server(options=limits.server_options)
    method = grpc.unary_unary_rpc_method_handler(
        handler,
        request_deserializer=lambda value: value,
        response_serializer=lambda value: value,
    )
    service = grpc.method_handlers_generic_handler(
        "ek.worker.v2.ComputationService",
        {"Execute": method},
    )
    server.add_generic_rpc_handlers((service,))
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    await server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        await server.stop(None)


def prepare_output(client: GrpcWorkerTransport) -> torch.Tensor:
    del client
    return torch.empty((2, 3), dtype=torch.float16)


def run(coroutine: Awaitable[None]) -> None:
    asyncio.run(coroutine)


def test_client_compacts_request_and_fills_preallocated_output() -> None:
    async def scenario() -> None:
        received: list[WorkerBatch] = []

        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            batch = decode_request(payload, batch_spec())
            received.append(batch)
            return encode_success_response(batch.hidden_states * 2, batch_spec())

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=2, device="cpu")
            await client.start()
            output = prepare_output(client)
            await client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )

            torch.testing.assert_close(
                output,
                torch.tensor([[14.0, 16.0, 18.0], [2.0, 4.0, 6.0]], dtype=torch.float16),
            )
            assert len(received) == 1
            assert received[0].token_indices is None
            assert received[0].expert_ids.tolist() == [[1, -1], [0, 3]]
            await client.close()

    run(scenario())


def test_client_maps_structured_computation_error() -> None:
    async def scenario() -> None:
        busy = TransportError(
            TransportErrorCode.BUSY,
            retryable=True,
            diagnostic="waiting area is full",
        )

        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            return encode_error_response(busy, batch_spec())

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)

            with pytest.raises(TransportError) as caught:
                await client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )

            assert caught.value.code is TransportErrorCode.BUSY
            assert caught.value.retryable is True
            assert caught.value.diagnostic == "waiting area is full"
            await client.close()

    run(scenario())


def test_client_rejects_endpoint_shape_mismatch_before_rpc() -> None:
    async def scenario() -> None:
        calls = 0

        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            nonlocal calls
            calls += 1
            raise AssertionError("invalid local batch must not reach the Worker")

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)
            source = worker_batch()
            wrong_instance = WorkerBatch(
                instance_id=8,
                layer_id=source.layer_id,
                topology_version=source.topology_version,
                hidden_states=source.hidden_states,
                token_indices=source.token_indices,
                expert_ids=source.expert_ids,
                routing_weights=source.routing_weights,
                distinct_expert_ids=source.distinct_expert_ids,
            )

            with pytest.raises(TransportError) as caught:
                await client.execute(
                    wrong_instance,
                    output,
                    monotonic_deadline=float("inf"),
                )

            assert caught.value.code is TransportErrorCode.INVALID_REQUEST
            assert caught.value.retryable is False
            assert calls == 0
            await client.close()

    run(scenario())


@pytest.mark.parametrize(
    ("status", "expected_code", "retryable"),
    [
        (grpc.StatusCode.RESOURCE_EXHAUSTED, TransportErrorCode.BUSY, True),
        (grpc.StatusCode.INVALID_ARGUMENT, TransportErrorCode.PROTOCOL, False),
        (grpc.StatusCode.UNAVAILABLE, TransportErrorCode.UNAVAILABLE, True),
        (grpc.StatusCode.UNIMPLEMENTED, TransportErrorCode.UNSUPPORTED, False),
    ],
)
def test_client_maps_native_grpc_status(
    status: grpc.StatusCode,
    expected_code: TransportErrorCode,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            await context.abort(status, "test status")
            raise AssertionError("abort must terminate the handler")

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)

            with pytest.raises(TransportError) as caught:
                await client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )

            assert caught.value.code is expected_code
            assert caught.value.retryable is retryable
            await client.close()

    run(scenario())


def test_client_bounds_calls_before_cpu_encoding_and_rpc() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        two_started = asyncio.Event()
        calls = 0

        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 2:
                two_started.set()
            await gate.wait()
            batch = decode_request(payload, batch_spec())
            return encode_success_response(batch.hidden_states, batch_spec())

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=2, device="cpu")
            await client.start()
            outputs = [prepare_output(client) for _ in range(3)]
            submissions = [
                asyncio.create_task(
                    client.execute(
                        worker_batch(),
                        output,
                        monotonic_deadline=float("inf"),
                    )
                )
                for output in outputs
            ]
            async with asyncio.timeout(2):
                await two_started.wait()

            assert calls == 2
            gate.set()
            await asyncio.gather(*submissions)
            assert calls == 3
            await client.close()

    run(scenario())


def test_client_uses_remaining_deadline_for_native_rpc() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("deadline must cancel the handler")

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)
            submission = asyncio.create_task(
                client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 0.2,
                )
            )
            async with asyncio.timeout(2):
                await started.wait()

            with pytest.raises(TransportError) as caught:
                await submission
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED
            assert caught.value.retryable is False
            await client.close()

    run(scenario())


def test_client_cancellation_waits_for_adapter_cleanup_and_allows_reuse() -> None:
    async def scenario() -> None:
        first_started = asyncio.Event()
        first_gate = asyncio.Event()
        call_count = 0

        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            nonlocal call_count
            call_count += 1
            batch = decode_request(payload, batch_spec())
            if call_count == 1:
                first_started.set()
                await first_gate.wait()
            return encode_success_response(batch.hidden_states, batch_spec())

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)
            submission = asyncio.create_task(
                client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )
            )
            await first_started.wait()
            submission.cancel()

            with pytest.raises(asyncio.CancelledError):
                await submission
            first_gate.set()
            await client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
            assert call_count == 2
            await client.close()

    run(scenario())


def test_client_rejects_malformed_success_response() -> None:
    async def scenario() -> None:
        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            return b"\xff"

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)

            with pytest.raises(TransportError) as caught:
                await client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )
            assert caught.value.code is TransportErrorCode.PROTOCOL
            assert caught.value.retryable is False
            await client.close()

    run(scenario())


def test_client_close_is_idempotent_and_stops_new_submissions() -> None:
    async def scenario() -> None:
        async def execute(payload: bytes, context: grpc.aio.ServicerContext) -> bytes:
            batch = decode_request(payload, batch_spec())
            return encode_success_response(batch.hidden_states, batch_spec())

        async with raw_server(execute) as endpoint:
            client = GrpcWorkerTransport(endpoint, batch_spec(), max_in_flight=1, device="cpu")
            await client.start()
            output = prepare_output(client)
            await client.close()
            await client.close()

            with pytest.raises(TransportError) as caught:
                await client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )
            assert caught.value.code is TransportErrorCode.UNAVAILABLE
            assert caught.value.retryable is True

    run(scenario())
