"""Integration tests for the bounded Worker-side gRPC receiver."""

import asyncio
import time
from collections.abc import Awaitable

import grpc
import pytest
import torch

from expertkit_transport.adapters.grpc import (
    GrpcBatchSpec,
    GrpcWorkerServer,
    GrpcWorkerTransport,
)
from expertkit_transport.adapters.grpc.spec import calculate_message_limits
from expertkit_transport.contracts import (
    OutputSpec,
    ReceiverClosed,
    TransportError,
    TransportErrorCode,
    WorkerBatch,
)


def batch_spec() -> GrpcBatchSpec:
    return GrpcBatchSpec(
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


def prepare_output(client: GrpcWorkerTransport):
    return client.output_buffers.prepare(OutputSpec(4, 3, torch.float16, "cpu"))


async def start_pair(
    *,
    max_active_batches: int = 1,
    max_pending_batches: int = 1,
    client_in_flight: int = 2,
) -> tuple[GrpcWorkerServer, GrpcWorkerTransport]:
    server = GrpcWorkerServer(
        "127.0.0.1:0",
        batch_spec(),
        max_active_batches=max_active_batches,
        max_pending_batches=max_pending_batches,
    )
    await server.start()
    client = GrpcWorkerTransport(
        f"127.0.0.1:{server.bound_port}",
        batch_spec(),
        max_in_flight=client_in_flight,
    )
    await client.start()
    return server, client


async def close_pair(server: GrpcWorkerServer, client: GrpcWorkerTransport) -> None:
    await client.close()
    await server.close()


def run(coroutine: Awaitable[None]) -> None:
    asyncio.run(coroutine)


def test_server_hands_one_validated_batch_directly_to_execution() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.submit(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )

        received = await server.take()
        assert received.batch.token_indices is None
        torch.testing.assert_close(
            received.batch.hidden_states,
            torch.tensor([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]], dtype=torch.float16),
        )
        assert server.pending_count == 0
        assert server.active_count == 1
        assert server.admitted_count(2, 0) == 1
        assert server.admitted_count(2, 1) == 1
        assert server.admitted_count(2, 3) == 1

        idle = asyncio.create_task(
            server.wait_experts_idle(
                ((2, 0), (2, 1), (2, 3)), monotonic_deadline=time.monotonic() + 2
            )
        )
        await received.complete(received.batch.hidden_states * 2)
        await submission
        await idle

        torch.testing.assert_close(
            output.tensor[:2],
            torch.tensor([[14.0, 16.0, 18.0], [2.0, 4.0, 6.0]], dtype=torch.float16),
        )
        assert server.active_count == 0
        assert server.admitted_count(2, 0) == 0
        client.output_buffers.release(output)
        await close_pair(server, client)

    run(scenario())


def test_server_delivers_structured_execution_rejection() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.submit(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        received = await server.take()
        not_ready = TransportError(
            TransportErrorCode.EXPERT_NOT_READY,
            retryable=True,
            unavailable_expert_ids=(3,),
            diagnostic="expert is not ready",
        )

        await received.reject(not_ready)
        with pytest.raises(TransportError) as caught:
            await submission

        assert caught.value.code is TransportErrorCode.EXPERT_NOT_READY
        assert caught.value.unavailable_expert_ids == (3,)
        assert server.active_count == 0
        client.output_buffers.release(output)
        await close_pair(server, client)

    run(scenario())


def test_application_pending_limit_returns_busy_and_cancellation_releases_input() -> None:
    async def scenario() -> None:
        server, client = await start_pair(client_in_flight=2)
        first_output = prepare_output(client)
        second_output = prepare_output(client)
        first = asyncio.create_task(
            client.submit(
                worker_batch(),
                first_output,
                monotonic_deadline=float("inf"),
            )
        )
        async with asyncio.timeout(2):
            await server._wait_pending_count(1)

        with pytest.raises(TransportError) as caught:
            await client.submit(
                worker_batch(),
                second_output,
                monotonic_deadline=float("inf"),
            )
        assert caught.value.code is TransportErrorCode.BUSY
        assert caught.value.retryable is True
        assert server.pending_count == 1
        assert server.pending_retained_bytes > 0

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        async with asyncio.timeout(2):
            await server._wait_pending_count(0)
        assert server.pending_retained_bytes == 0
        assert server.admitted_count(2, 0) == 0

        client.output_buffers.release(first_output)
        client.output_buffers.release(second_output)
        await close_pair(server, client)

    run(scenario())


def test_outer_grpc_concurrency_limit_bounds_active_plus_pending_calls() -> None:
    async def scenario() -> None:
        server, client = await start_pair(client_in_flight=3)
        outputs = [prepare_output(client) for _ in range(3)]
        first = asyncio.create_task(
            client.submit(
                worker_batch(),
                outputs[0],
                monotonic_deadline=float("inf"),
            )
        )
        active = await server.take()
        others = [
            asyncio.create_task(
                client.submit(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )
            )
            for output in outputs[1:]
        ]
        second = await server.take()

        await active.complete(active.batch.hidden_states)
        await second.complete(second.batch.hidden_states)
        results = await asyncio.gather(first, *others, return_exceptions=True)

        failures = [result for result in results if isinstance(result, TransportError)]
        assert len(failures) == 1
        assert failures[0].code is TransportErrorCode.BUSY
        assert server.active_count == 0
        for output in outputs:
            client.output_buffers.release(output)
        await close_pair(server, client)

    run(scenario())


def test_cancellation_after_take_discards_response_but_releases_expert_use() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.submit(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        received = await server.take()
        submission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submission
        async with asyncio.timeout(2):
            await server._wait_cancelled(received)

        assert received.cancelled is True
        assert server.active_count == 1
        await received.complete(received.batch.hidden_states)
        assert server.active_count == 0
        assert server.admitted_count(2, 1) == 0
        client.output_buffers.release(output)
        await close_pair(server, client)

    run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"",
    ],
)
def test_server_rejects_malformed_wire_requests_with_native_status(payload: bytes) -> None:
    async def scenario() -> None:
        server = GrpcWorkerServer(
            "127.0.0.1:0",
            batch_spec(),
            max_active_batches=1,
            max_pending_batches=1,
        )
        await server.start()
        limits = calculate_message_limits(batch_spec())
        channel = grpc.aio.insecure_channel(
            f"127.0.0.1:{server.bound_port}",
            options=limits.client_options,
        )
        execute = channel.unary_unary(
            "/ek.worker.v2.ComputationService/Execute",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )

        with pytest.raises(grpc.aio.AioRpcError) as caught:
            await execute(payload)
        assert caught.value.code() is grpc.StatusCode.INVALID_ARGUMENT
        assert server.pending_count == 0
        await channel.close()
        await server.close()

    run(scenario())


def test_server_close_wakes_execution_waiting_on_take() -> None:
    async def scenario() -> None:
        server = GrpcWorkerServer(
            "127.0.0.1:0",
            batch_spec(),
            max_active_batches=1,
            max_pending_batches=1,
        )
        await server.start()
        waiting = asyncio.create_task(server.take())
        await server.close()
        await server.close()

        with pytest.raises(ReceiverClosed):
            await waiting

    run(scenario())
