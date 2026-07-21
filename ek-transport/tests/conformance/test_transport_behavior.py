"""Run common request behavior against every current Transport implementation."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, replace
from typing import Literal

import grpc
import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerBatchReceiver,
    WorkerEndpointConfig,
    WorkerTransport,
)
from expertkit_transport.transports.grpc import (
    GrpcWorkerBatchReceiver,
    GrpcWorkerTransport,
)
from expertkit_transport.transports.shm import (
    ShmWorkerBatchReceiver,
    ShmWorkerTransport,
)

from .in_memory_transport import create_in_memory_pair

TransportKind = Literal["grpc", "shm", "in_memory"]
_KINDS: tuple[TransportKind, ...] = ("grpc", "shm", "in_memory")


def endpoint_config() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float32,
    )


def worker_batch() -> WorkerBatch:
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


def received_batch() -> WorkerBatch:
    source = worker_batch()
    assert source.token_indices is not None
    return WorkerBatch(
        instance_id=source.instance_id,
        layer_id=source.layer_id,
        topology_version=source.topology_version,
        hidden_states=source.hidden_states.index_select(0, source.token_indices),
        token_indices=None,
        expert_ids=source.expert_ids,
        routing_weights=source.routing_weights,
        distinct_expert_ids=source.distinct_expert_ids,
    )


@dataclass(slots=True)
class RunningPair:
    kind: TransportKind
    receiver: WorkerBatchReceiver
    transport: WorkerTransport
    buffers: WorkerBatchBuffers
    endpoint: str | None

    async def close(self) -> None:
        try:
            await self.transport.close()
        finally:
            try:
                await self.receiver.close()
            finally:
                self.buffers.close()


async def start_pair(kind: TransportKind) -> RunningPair:
    config = endpoint_config()
    buffer_config = BatchBufferConfig(4, 3, 2, torch.float32, "cpu")
    endpoint: str | None = None
    if kind == "grpc":
        receiver = GrpcWorkerBatchReceiver(
            "127.0.0.1:0",
            config,
            max_active_batches=1,
            max_pending_batches=1,
        )
        buffers = receiver.create_batch_buffers(buffer_config)
        await receiver.start()
        endpoint = f"127.0.0.1:{receiver.bound_port}"
        transport = GrpcWorkerTransport(
            endpoint,
            config,
            max_in_flight=2,
            device="cpu",
        )
    elif kind == "shm":
        receiver = ShmWorkerBatchReceiver(
            "127.0.0.1:0",
            config,
            max_active_batches=1,
            max_pending_batches=1,
        )
        buffers = receiver.create_batch_buffers(buffer_config)
        await receiver.start()
        endpoint = f"127.0.0.1:{receiver.bound_port}"
        transport = ShmWorkerTransport(
            endpoint,
            config,
            max_in_flight=2,
            device="cpu",
        )
    else:
        receiver, transport = create_in_memory_pair(config, max_pending_batches=1)
        buffers = receiver.create_batch_buffers(buffer_config)
        await receiver.start()
    await transport.start()
    return RunningPair(kind, receiver, transport, buffers, endpoint)


async def finish_success(received: ReceivedBatch) -> torch.Tensor:
    source = received.batch
    partial_output = source.hidden_states * 2
    destination = received.output_destination
    if destination is not None:
        destination.copy_(partial_output)
        partial_output = destination
    received.release_input()
    await received.complete(partial_output)
    return partial_output


async def wait_pending(pair: RunningPair) -> None:
    async with asyncio.timeout(2):
        while pair.receiver.pending_count != 1:  # type: ignore[attr-defined]
            await asyncio.sleep(0)


@pytest.mark.parametrize("kind", _KINDS)
def test_success_preserves_input_and_output_lifetimes(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        batch = worker_batch()
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            submission = asyncio.create_task(
                pair.transport.execute(
                    batch,
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await pair.receiver.receive()
            batch.hidden_states.fill_(99)
            assert received.batch.hidden_states.tolist() == [[7, 8, 9], [1, 2, 3]]
            partial_output = await finish_success(received)
            await submission
            expected = torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32)
            torch.testing.assert_close(output, expected)
            partial_output.fill_(-1)
            torch.testing.assert_close(output, expected)
        finally:
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_output_shape_and_dtype_are_validated(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        try:
            with pytest.raises(ValueError, match="shape"):
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((1, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            with pytest.raises(ValueError, match="dtype"):
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float16),
                    monotonic_deadline=time.monotonic() + 5,
                )
        finally:
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_batch_token_limit_is_enforced(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        oversized = WorkerBatch(
            instance_id=7,
            layer_id=2,
            topology_version=11,
            hidden_states=torch.ones((5, 3), dtype=torch.float32),
            token_indices=None,
            expert_ids=torch.zeros((5, 2), dtype=torch.int32),
            routing_weights=torch.ones((5, 2), dtype=torch.float32),
            distinct_expert_ids=(0,),
        )
        try:
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    oversized,
                    torch.empty((5, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.INVALID_REQUEST
        finally:
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_pending_limit_returns_busy(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        first_output = torch.empty((2, 3), dtype=torch.float32)
        first = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                first_output,
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            await wait_pending(pair)
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.BUSY
            received = await pair.receiver.receive()
            await finish_success(received)
            await first
        finally:
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_drain_clear_and_idle_wait_share_one_contract(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            await pair.receiver.begin_drain(
                ((2, 1),),
                min_topology_version=17,
                stop_all=False,
            )
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.DRAINING
            assert caught.value.min_topology_version == 17

            await pair.receiver.clear_drains(((2, 1),))
            submission = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            await wait_pending(pair)
            with pytest.raises(TimeoutError):
                await pair.receiver.wait_idle(
                    ((2, 1),),
                    monotonic_deadline=0.0,
                )
            received = await pair.receiver.receive()
            await finish_success(received)
            await submission
            await pair.receiver.wait_idle(((2, 1),), monotonic_deadline=math.inf)
            await pair.receiver.wait_idle(None, monotonic_deadline=math.inf)
        finally:
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_cancellation_keeps_receiver_ownership_until_completion(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            received = await pair.receiver.receive()
            submission.cancel()
            if kind == "shm":
                await asyncio.sleep(0)
                assert not submission.done()
            else:
                async with asyncio.timeout(2):
                    while not received.cancelled:
                        await asyncio.sleep(0)
            assert pair.receiver.active_count == 1  # type: ignore[attr-defined]
            await finish_success(received)
            with pytest.raises(asyncio.CancelledError):
                await submission
            await pair.receiver.wait_idle(None, monotonic_deadline=math.inf)
        finally:
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_receiver_shutdown_waits_for_active_completion(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        close_task: asyncio.Task[None] | None = None
        try:
            received = await pair.receiver.receive()
            close_task = asyncio.create_task(pair.receiver.close())
            await asyncio.sleep(0)
            assert not close_task.done()
            await finish_success(received)
            await close_task
            await asyncio.gather(submission, return_exceptions=True)
        finally:
            if close_task is not None and not close_task.done():
                close_task.cancel()
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await pair.transport.close()
            pair.buffers.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_expired_deadline_and_closed_peer_fail_without_admission(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=0.0,
                )
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED

            await pair.receiver.close()
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 1,
                )
            assert caught.value.code is TransportErrorCode.UNAVAILABLE
        finally:
            await pair.transport.close()
            pair.buffers.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_fixed_worker_buffers_are_reused(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        hidden = torch.empty((2, 3), dtype=torch.float32)
        expert_ids = torch.empty((2, 2), dtype=torch.int32)
        routing_weights = torch.empty((2, 2), dtype=torch.float32)
        output = torch.empty((2, 3), dtype=torch.float32)
        pointers = tuple(
            tensor.data_ptr() for tensor in (hidden, expert_ids, routing_weights, output)
        )
        try:
            for _ in range(2):
                pair.buffers.copy_input(
                    received_batch(),
                    hidden,
                    expert_ids,
                    routing_weights,
                )
                destination = output if kind == "shm" else None
                returned = pair.buffers.copy_output(output, destination)
                assert returned.data_ptr() == output.data_ptr()
                assert pointers == tuple(
                    tensor.data_ptr() for tensor in (hidden, expert_ids, routing_weights, output)
                )
        finally:
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ("grpc", "shm"))
def test_malformed_wire_message_is_rejected(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        assert pair.endpoint is not None
        channel = grpc.aio.insecure_channel(pair.endpoint)
        method = (
            "/ek.worker.v2.ComputationService/Execute"
            if kind == "grpc"
            else "/ek.worker.v2.ComputationService/ExecuteSharedMemory"
        )
        call = channel.unary_unary(
            method,
            request_serializer=lambda payload: payload,
            response_deserializer=lambda payload: payload,
        )
        try:
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await call(b"\xff")
            assert caught.value.code() is grpc.StatusCode.INVALID_ARGUMENT
        finally:
            await channel.close()
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", _KINDS)
def test_receiver_revalidates_routing_values(kind: TransportKind) -> None:
    async def scenario() -> None:
        pair = await start_pair(kind)
        invalid = replace(
            worker_batch(),
            expert_ids=torch.tensor([[-1, 1], [0, 3]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        )
        try:
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    invalid,
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code in {
                TransportErrorCode.INVALID_REQUEST,
                TransportErrorCode.PROTOCOL,
            }
        finally:
            await pair.close()

    asyncio.run(scenario())
