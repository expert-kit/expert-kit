"""NCCL Transport behavior over an injectable CPU runtime."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerEndpointConfig,
)
from expertkit_transport.transports.nccl import (
    NcclWorkerBatchReceiver,
    NcclWorkerTransport,
)


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


def worker_batch(*, offset: float = 0) -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor(
            [[1 + offset, 2 + offset, 3 + offset], [4, 5, 6], [7, 8, 9]],
            dtype=torch.float32,
        ),
        token_indices=torch.tensor([2, 0], dtype=torch.int64),
        expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0], [0.5, 0.5]], dtype=torch.float32),
        distinct_expert_ids=(0, 1, 3),
    )


@dataclass(slots=True)
class _Transfer:
    sequence: int
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    response: asyncio.Future[torch.Tensor]


class _FakeWorkerExchange:
    def __init__(self, transfer: _Transfer, dummy_output: torch.Tensor) -> None:
        self._transfer = transfer
        self._dummy_output = dummy_output
        self._finished = False

    async def send_output(
        self,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        del monotonic_deadline
        if self._finished:
            raise RuntimeError("fake NCCL exchange is already finished")
        self._finished = True
        if not self._transfer.response.done():
            self._transfer.response.set_result(output.detach().clone())

    async def abort(self, *, monotonic_deadline: float) -> None:
        del monotonic_deadline
        if self._finished:
            return
        self._finished = True
        if not self._transfer.response.done():
            self._transfer.response.set_result(self._dummy_output.detach().clone())


class _FakeNcclLink:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[_Transfer] = asyncio.Queue()
        self.exchange_sequences: list[int] = []
        self.exchange_changed = asyncio.Condition()

    async def record_exchange(self, transfer: _Transfer) -> None:
        async with self.exchange_changed:
            self.exchange_sequences.append(transfer.sequence)
            await self.queue.put(transfer)
            self.exchange_changed.notify_all()

    async def wait_exchange_count(self, count: int) -> None:
        async with asyncio.timeout(2), self.exchange_changed:
            await self.exchange_changed.wait_for(lambda: len(self.exchange_sequences) >= count)


class _FakeNcclRuntime:
    def __init__(self, link: _FakeNcclLink, *, rank: int) -> None:
        self._link = link
        self._rank = rank
        self._started = False
        self._closed = False
        self._next_sequence = 1
        self._peer_lock = asyncio.Lock()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return 2

    @property
    def group_name(self) -> str:
        return "fake-nccl"

    @property
    def rendezvous_endpoint(self) -> str:
        return "fake://nccl-tests"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("fake NCCL runtime is closed")
        self._started = True

    async def wait_ready(self, *, monotonic_deadline: float) -> None:
        if monotonic_deadline <= time.monotonic():
            raise TimeoutError("fake NCCL readiness deadline expired")
        if not self._started:
            raise RuntimeError("fake NCCL runtime has not been started")

    async def exchange(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        assert self._rank == 0
        assert peer_rank == 1
        async with self._peer_lock:
            loop = asyncio.get_running_loop()
            transfer = _Transfer(
                sequence=self._next_sequence,
                hidden_states=hidden_states.detach().clone(),
                expert_ids=expert_ids.detach().clone(),
                routing_weights=routing_weights.detach().clone(),
                response=loop.create_future(),
            )
            self._next_sequence += 1
            await self._link.record_exchange(transfer)
            cancelled = False
            try:
                result = await asyncio.shield(transfer.response)
            except asyncio.CancelledError:
                cancelled = True
                result = await asyncio.shield(transfer.response)
            output.copy_(result)
            if cancelled:
                raise asyncio.CancelledError
            if monotonic_deadline <= time.monotonic():
                raise TransportError(
                    TransportErrorCode.DEADLINE_EXCEEDED,
                    retryable=False,
                    diagnostic="fake NCCL deadline expired after matching",
                )

    async def receive_inputs(
        self,
        peer_rank: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        dummy_output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> _FakeWorkerExchange:
        assert self._rank == 1
        assert peer_rank == 0
        del monotonic_deadline
        transfer = await self._link.queue.get()
        token_count = transfer.hidden_states.shape[0]
        hidden_states[:token_count].copy_(transfer.hidden_states)
        expert_ids[:token_count].copy_(transfer.expert_ids)
        routing_weights[:token_count].copy_(transfer.routing_weights)
        return _FakeWorkerExchange(transfer, dummy_output)

    async def close(self) -> None:
        self._closed = True


@dataclass(slots=True)
class _RunningPair:
    receiver: NcclWorkerBatchReceiver
    transport: NcclWorkerTransport
    buffers: WorkerBatchBuffers
    link: _FakeNcclLink
    client_runtime: _FakeNcclRuntime
    worker_runtime: _FakeNcclRuntime

    async def close(self) -> None:
        try:
            await self.transport.close()
        finally:
            try:
                await self.receiver.close()
            finally:
                self.buffers.close()
                await self.client_runtime.close()
                await self.worker_runtime.close()


async def start_pair(
    *,
    max_active_batches: int = 1,
    max_pending_batches: int = 1,
    max_in_flight: int = 2,
    admission_delay_seconds: float = 0,
) -> _RunningPair:
    link = _FakeNcclLink()
    client_runtime = _FakeNcclRuntime(link, rank=0)
    worker_runtime = _FakeNcclRuntime(link, rank=1)
    config = endpoint_config()
    receiver = NcclWorkerBatchReceiver(
        "127.0.0.1:0",
        config,
        runtime=worker_runtime,
        max_active_batches=max_active_batches,
        max_pending_batches=max_pending_batches,
        owns_runtime=False,
    )
    if admission_delay_seconds:
        execute = receiver._execute

        async def delay_first_response(payload: bytes, context: object):
            first = True
            async for response in execute(payload, context):  # type: ignore[arg-type]
                if first:
                    first = False
                    await asyncio.sleep(admission_delay_seconds)
                yield response

        receiver._execute = delay_first_response  # type: ignore[method-assign]
    buffers = receiver.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
    await receiver.start()
    transport = NcclWorkerTransport(
        f"127.0.0.1:{receiver.bound_port}",
        config,
        max_in_flight=max_in_flight,
        device="cpu",
        runtime=client_runtime,
        peer_rank=1,
    )
    await transport.start()
    return _RunningPair(
        receiver,
        transport,
        buffers,
        link,
        client_runtime,
        worker_runtime,
    )


async def complete(received: ReceivedBatch, *, multiplier: float = 2) -> torch.Tensor:
    partial_output = received.batch.hidden_states * multiplier
    destination = received.output_destination
    assert destination is not None
    destination.copy_(partial_output)
    partial_output = destination
    received.release_input()
    await received.complete(partial_output)
    return partial_output


async def wait_pending(pair: _RunningPair, count: int = 1) -> None:
    async with asyncio.timeout(2):
        while pair.receiver.pending_count != count:
            await asyncio.sleep(0)


def test_success_preserves_caller_and_worker_tensor_lifetimes() -> None:
    async def scenario() -> None:
        pair = await start_pair()
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
            partial_output = await complete(received)
            await submission

            expected = torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32)
            torch.testing.assert_close(output, expected)
            partial_output.fill_(-1)
            torch.testing.assert_close(output, expected)
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_admission_rejection_happens_before_a_second_nccl_exchange() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_pending_batches=1, max_in_flight=2)
        contender = NcclWorkerTransport(
            f"127.0.0.1:{pair.receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=pair.client_runtime,
            peer_rank=1,
        )
        await contender.start()
        first = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            await wait_pending(pair)
            await pair.link.wait_exchange_count(1)
            with pytest.raises(TransportError) as caught:
                await contender.execute(
                    worker_batch(offset=10),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.BUSY
            assert pair.link.exchange_sequences == [1]

            received = await pair.receiver.receive()
            await complete(received)
            await first
        finally:
            if not first.done():
                first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await contender.close()
            await pair.close()

    asyncio.run(scenario())


def test_concurrent_calls_submit_nccl_exchanges_in_peer_order() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_pending_batches=2, max_in_flight=2)
        first_output = torch.empty((2, 3), dtype=torch.float32)
        second_output = torch.empty((2, 3), dtype=torch.float32)
        first = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                first_output,
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        second: asyncio.Task[None] | None = None
        try:
            await pair.link.wait_exchange_count(1)
            first_received = await pair.receiver.receive()
            second = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(offset=10),
                    second_output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            await asyncio.sleep(0.05)
            assert pair.link.exchange_sequences == [1]

            await complete(first_received)
            await first
            await pair.link.wait_exchange_count(2)
            second_received = await pair.receiver.receive()
            await complete(second_received)
            await second

            assert pair.link.exchange_sequences == [1, 2]
            torch.testing.assert_close(
                first_output,
                torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32),
            )
            torch.testing.assert_close(
                second_output,
                torch.tensor([[14, 16, 18], [22, 24, 26]], dtype=torch.float32),
            )
        finally:
            for submission in (first, second):
                if submission is not None and not submission.done():
                    submission.cancel()
            await asyncio.gather(
                *(task for task in (first, second) if task is not None),
                return_exceptions=True,
            )
            await pair.close()

    asyncio.run(scenario())


def test_cancellation_waits_for_matching_nccl_completion_before_reuse() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
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
            await asyncio.sleep(0)
            assert not submission.done()
            assert pair.receiver.active_count == 1

            await complete(received)
            with pytest.raises(asyncio.CancelledError):
                await submission
            await pair.receiver.wait_idle(None, monotonic_deadline=math.inf)

            retry_output = torch.empty((2, 3), dtype=torch.float32)
            retry = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(),
                    retry_output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            retried = await pair.receiver.receive()
            await complete(retried)
            await retry
        finally:
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await pair.close()

    asyncio.run(scenario())


def test_inflight_deadline_waits_for_matching_nccl_completion_before_reuse() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 0.1,
            )
        )
        try:
            received = await pair.receiver.receive()
            await asyncio.sleep(0.15)
            assert not submission.done()

            await complete(received)
            with pytest.raises(TransportError) as caught:
                await submission
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED
            await pair.receiver.wait_idle(None, monotonic_deadline=math.inf)

            retry_output = torch.empty((2, 3), dtype=torch.float32)
            retry = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(),
                    retry_output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            retried = await pair.receiver.receive()
            await complete(retried)
            await retry
        finally:
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await pair.close()

    asyncio.run(scenario())


def test_deadline_at_admission_delivery_still_closes_the_nccl_exchange() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1, admission_delay_seconds=0.15)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 0.05,
            )
        )
        try:
            async with asyncio.timeout(2):
                received = await pair.receiver.receive()
            assert pair.link.exchange_sequences == [1]
            assert not submission.done()

            await complete(received)
            with pytest.raises(TransportError) as caught:
                await submission
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED
            await pair.receiver.wait_idle(None, monotonic_deadline=math.inf)
        finally:
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await pair.close()

    asyncio.run(scenario())


def test_deadline_and_structured_rejection_leave_the_link_reusable() -> None:
    async def scenario() -> None:
        pair = await start_pair()
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=0,
                )
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED
            assert pair.link.exchange_sequences == []

            rejected = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await pair.receiver.receive()
            received.release_input()
            await received.reject(
                TransportError(
                    TransportErrorCode.EXPERT_NOT_READY,
                    retryable=True,
                    unavailable_expert_ids=(3,),
                    diagnostic="fixture expert is not ready",
                )
            )
            with pytest.raises(TransportError) as caught:
                await rejected
            assert caught.value.code is TransportErrorCode.EXPERT_NOT_READY
            assert caught.value.unavailable_expert_ids == (3,)

            retry = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            retried = await pair.receiver.receive()
            await complete(retried)
            await retry
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_receiver_close_waits_for_active_nccl_completion() -> None:
    async def scenario() -> None:
        pair = await start_pair()
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
            await complete(received)
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
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_transport_close_waits_for_inflight_exchange_and_stops_admission() -> None:
    async def scenario() -> None:
        pair = await start_pair()
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
            close_task = asyncio.create_task(pair.transport.close())
            await asyncio.sleep(0)
            assert not close_task.done()
            with pytest.raises(TransportError) as caught:
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert caught.value.code is TransportErrorCode.UNAVAILABLE

            await complete(received)
            await submission
            await close_task
        finally:
            if close_task is not None and not close_task.done():
                close_task.cancel()
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            await pair.transport.close()
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_fixed_worker_buffers_and_metadata_staging_are_reused() -> None:
    async def scenario() -> None:
        pair = await start_pair()
        hidden_states = torch.empty((2, 3), dtype=torch.float32)
        expert_ids = torch.empty((2, 2), dtype=torch.int32)
        routing_weights = torch.empty((2, 2), dtype=torch.float32)
        output = torch.empty((2, 3), dtype=torch.float32)
        destination = torch.empty_like(output)
        pointers = tuple(
            tensor.data_ptr()
            for tensor in (
                hidden_states,
                expert_ids,
                routing_weights,
                output,
                destination,
            )
        )
        received_batch = WorkerBatch(
            instance_id=7,
            layer_id=2,
            topology_version=11,
            hidden_states=torch.tensor([[7, 8, 9], [1, 2, 3]], dtype=torch.float32),
            token_indices=None,
            expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.25, 0], [0.5, 0.5]], dtype=torch.float32),
            distinct_expert_ids=(0, 1, 3),
        )
        try:
            for _ in range(2):
                pair.buffers.copy_input(
                    received_batch,
                    hidden_states,
                    expert_ids,
                    routing_weights,
                )
                returned = pair.buffers.copy_output(
                    output,
                    destination,
                )
                assert returned.data_ptr() == destination.data_ptr()
                assert pair.buffers.host_staging_bytes == 4 * 2 * 8
                assert pointers == tuple(
                    tensor.data_ptr()
                    for tensor in (
                        hidden_states,
                        expert_ids,
                        routing_weights,
                        output,
                        destination,
                    )
                )
        finally:
            await pair.close()

    asyncio.run(scenario())
