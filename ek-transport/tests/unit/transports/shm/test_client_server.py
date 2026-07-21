"""Integration tests for shared Tensor slots over real gRPC notifications."""

import asyncio
import time
from dataclasses import replace

import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.buffers.base import OutputSpec
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import WorkerPositionSpec
from expertkit_transport.transports.grpc import GrpcBatchSpec, GrpcWorkerServer
from expertkit_transport.transports.shm import ShmWorkerTransport


def spec() -> GrpcBatchSpec:
    return GrpcBatchSpec(
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


async def start_pair() -> tuple[GrpcWorkerServer, ShmWorkerTransport]:
    server = GrpcWorkerServer(
        "127.0.0.1:0",
        spec(),
        max_active_batches=1,
        max_pending_batches=1,
    )
    buffers = server.allocate_position_buffers(WorkerPositionSpec(4, 3, 2, torch.float32, "cpu"))
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


def prepare(client: ShmWorkerTransport):
    return client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu"))


def test_shared_memory_path_compacts_and_returns_without_tensor_payloads() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare(client)
        try:
            submission = asyncio.create_task(
                client.submit(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            async with asyncio.timeout(2):
                received = await server.take()
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
                output.tensor[:2],
                torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32),
            )
            assert server.pending_retained_bytes == 0
        finally:
            await client.close()
            client.output_buffers.release(output)
            await server.close()

    asyncio.run(scenario())


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
                await client.submit(
                    invalid,
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.PROTOCOL
            assert server.pending_count == 0
        finally:
            await client.close()
            client.output_buffers.release(output)
            await server.close()

    asyncio.run(scenario())


def test_cancellation_waits_for_worker_release_before_slot_reuse() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare(client)
        try:
            submission = asyncio.create_task(
                client.submit(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            received = await server.take()
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
                client.submit(batch(), output, monotonic_deadline=time.monotonic() + 5)
            )
            retried = await server.take()
            retry_destination = retried.output_destination
            assert retry_destination is not None
            retry_destination.copy_(retried.batch.hidden_states)
            retried.release_input()
            await retried.complete(retry_destination)
            await retry
        finally:
            await client.close()
            client.output_buffers.release(output)
            await server.close()

    asyncio.run(scenario())
