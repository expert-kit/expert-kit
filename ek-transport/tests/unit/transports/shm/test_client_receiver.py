"""Integration tests for shared Tensor slots over real gRPC notifications."""

import asyncio
import time
from contextlib import contextmanager
from dataclasses import replace

import grpc
import pytest
import torch

import expertkit_transport.transports.shm.client as shm_client_module
from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.base import BatchBufferConfig
from expertkit_transport.transports.shm import ShmWorkerTransport
from expertkit_transport.transports.shm.receiver import ShmWorkerBatchReceiver


def spec() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float32,
    )


def batch() -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            dtype=torch.float32,
        ),
        token_indices=torch.tensor([2, 0], dtype=torch.int64),
        expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0.0], [0.5, 0.5]], dtype=torch.float32),
        distinct_expert_ids=(0, 1, 3),
    )


async def start_pair() -> tuple[ShmWorkerBatchReceiver, ShmWorkerTransport]:
    server = ShmWorkerBatchReceiver(
        "127.0.0.1:0",
        spec(),
        max_active_batches=1,
        max_pending_batches=1,
    )
    buffers = server.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
    buffers.close()
    await server.start()
    client = ShmWorkerTransport(
        f"127.0.0.1:{server.bound_port}",
        spec(),
        max_in_flight=2,
        device="cpu",
    )
    await client.start()
    return server, client


def prepare(client: ShmWorkerTransport) -> torch.Tensor:
    del client
    return torch.empty((2, 3), dtype=torch.float32)


def test_shared_memory_path_compacts_and_returns_without_tensor_payloads() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare(client)
        try:
            submission = asyncio.create_task(
                client.execute(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            async with asyncio.timeout(2):
                received = await server.receive()
            source = received.batch
            assert source.token_indices is None
            assert source.expert_ids.tolist() == [[1, -1], [0, 3]]
            destination = received.output_destination
            assert destination is not None
            destination.copy_(source.hidden_states * 2)
            received.release_input()
            await received.complete(destination)
            await submission

            torch.testing.assert_close(
                output,
                torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32),
            )
            assert server.pending_retained_bytes == 0
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_shared_memory_records_stages_and_propagates_traceparent(monkeypatch) -> None:
    stages: list[str] = []
    span_attributes: dict[str, dict[str, object]] = {}
    metadata: dict[str, str] = {}

    @contextmanager
    def record_span(_tracer: object, name: str, **kwargs: object):
        stages.append(name)
        span_attributes[name] = dict(kwargs.get("attributes") or {})  # type: ignore[arg-type]
        yield None

    monkeypatch.setattr(shm_client_module, "get_frontend_tracer", object)
    monkeypatch.setattr(shm_client_module, "trace_span", record_span)
    monkeypatch.setattr(
        shm_client_module,
        "current_trace_metadata",
        lambda: (("traceparent", "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"),),
    )
    original_execute = ShmWorkerBatchReceiver._execute_shared_memory

    async def capture(
        receiver: ShmWorkerBatchReceiver,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        metadata.update(dict(context.invocation_metadata()))
        return await original_execute(receiver, payload, context)

    monkeypatch.setattr(ShmWorkerBatchReceiver, "_execute_shared_memory", capture)

    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare(client)
        try:
            submission = asyncio.create_task(
                client.execute(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            received = await server.receive()
            destination = received.output_destination
            assert destination is not None
            destination.copy_(received.batch.hidden_states)
            received.release_input()
            await received.complete(destination)
            await submission
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())

    assert stages == [
        "frontend.tensor_slice",
        "frontend.serialize",
        "frontend.send_worker",
        "frontend.deserialize",
    ]
    assert span_attributes["frontend.send_worker"]["expertkit.transport"] == "shm"
    assert metadata["traceparent"].startswith("00-0123456789abcdef")


def test_worker_revalidates_routing_values_from_shared_memory() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare(client)
        invalid = replace(
            batch(),
            expert_ids=torch.tensor([[-1, 1], [0, 3]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        )
        try:
            with pytest.raises(TransportError) as caught:
                await client.execute(
                    invalid,
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.PROTOCOL
            assert server.pending_count == 0
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_cancellation_waits_for_worker_release_before_slot_reuse() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare(client)
        try:
            submission = asyncio.create_task(
                client.execute(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            received = await server.receive()
            submission.cancel()
            await asyncio.sleep(0)
            assert not submission.done()

            destination = received.output_destination
            assert destination is not None
            destination.copy_(received.batch.hidden_states)
            received.release_input()
            await received.complete(destination)
            with pytest.raises(asyncio.CancelledError):
                await submission

            retry = asyncio.create_task(
                client.execute(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            retried = await server.receive()
            retry_destination = retried.output_destination
            assert retry_destination is not None
            retry_destination.copy_(retried.batch.hidden_states)
            retried.release_input()
            await retried.complete(retry_destination)
            await retry
        finally:
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_control_close_is_not_blocked_by_data_rpc_capacity() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        first = asyncio.create_task(
            client.execute(
                batch(),
                prepare(client),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            await server._wait_pending_count(1)
            with pytest.raises(TransportError) as caught:
                await client.execute(
                    batch(),
                    prepare(client),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.BUSY

            received = await server.receive()
            destination = received.output_destination
            assert destination is not None
            destination.copy_(received.batch.hidden_states)
            received.release_input()
            await received.complete(destination)
            await first

            await client.close()
        finally:
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await client.close()
            await server.close()

    asyncio.run(scenario())


def test_shm_receiver_does_not_expose_grpc_tensor_execute() -> None:
    async def scenario() -> None:
        server = ShmWorkerBatchReceiver(
            "127.0.0.1:0",
            spec(),
            max_active_batches=1,
            max_pending_batches=1,
        )
        buffers = server.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
        buffers.close()
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
        execute = channel.unary_unary(
            "/ek.worker.v2.ComputationService/Execute",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await execute(b"")
            assert caught.value.code() is grpc.StatusCode.UNIMPLEMENTED
        finally:
            await channel.close()
            await server.close()

    asyncio.run(scenario())
