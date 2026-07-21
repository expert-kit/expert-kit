"""Integration tests for the bounded Worker-side gRPC receiver."""

import asyncio
import time
from collections.abc import Awaitable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace

import grpc
import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.tracing import TraceAttribute, TraceContext, Tracer, TraceSpan
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.base import BatchBufferConfig, ReceiverClosed
from expertkit_transport.transports.grpc import (
    GrpcWorkerBatchReceiver,
    GrpcWorkerTransport,
)
from expertkit_transport.transports.grpc.spec import calculate_message_limits


class RecordingSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, TraceAttribute] = {}
        self.ended = False

    def set_attribute(self, key: str, value: TraceAttribute) -> None:
        self.attributes[key] = value

    def is_recording(self) -> bool:
        return True

    def end(self) -> None:
        self.ended = True


class RecordingTracer:
    def __init__(self, *, recording: bool = True) -> None:
        self.spans: list[RecordingSpan] = []
        self.context = object()
        self.recording = recording

    def current_span_is_recording(self) -> bool:
        return self.recording

    def capture_context(self) -> TraceContext:
        return self.context

    def start_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> TraceSpan:
        span = RecordingSpan(name)
        span.attributes.update(attributes or {})
        self.spans.append(span)
        return span

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> Iterator[TraceSpan]:
        span = self.start_span(name, context=context, attributes=attributes)
        try:
            yield span
        finally:
            span.end()


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


def prepare_output(client: GrpcWorkerTransport) -> torch.Tensor:
    del client
    return torch.empty((2, 3), dtype=torch.float16)


async def start_pair(
    *,
    max_active_batches: int = 1,
    max_pending_batches: int = 1,
    client_in_flight: int = 2,
    pending_changes: list[int] | None = None,
    tracer: Tracer | None = None,
) -> tuple[GrpcWorkerBatchReceiver, GrpcWorkerTransport]:
    server = GrpcWorkerBatchReceiver(
        "127.0.0.1:0",
        batch_spec(),
        max_active_batches=max_active_batches,
        max_pending_batches=max_pending_batches,
        tracer=tracer,
        on_pending_changed=None if pending_changes is None else pending_changes.append,
    )
    await server.start()
    client = GrpcWorkerTransport(
        f"127.0.0.1:{server.bound_port}",
        batch_spec(),
        max_in_flight=client_in_flight,
        device="cpu",
    )
    await client.start()
    return server, client


async def close_pair(server: GrpcWorkerBatchReceiver, client: GrpcWorkerTransport) -> None:
    await client.close()
    await server.close()


def run(coroutine: Awaitable[None]) -> None:
    asyncio.run(coroutine)


async def await_with_loop_yields[T](awaitable: Awaitable[T]) -> T:
    """Keep the test loop runnable while executor threads finish codec work."""

    task = asyncio.ensure_future(awaitable)
    async with asyncio.timeout(2):
        while not task.done():
            await asyncio.sleep(0)
    return task.result()


def test_receiver_creates_direct_cpu_batch_buffers() -> None:
    server = GrpcWorkerBatchReceiver(
        "127.0.0.1:0",
        batch_spec(),
        max_active_batches=1,
        max_pending_batches=1,
    )
    buffers = server.create_batch_buffers(
        BatchBufferConfig(
            max_batch_tokens=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float16,
            device="cpu",
        )
    )
    batch = worker_batch()
    compact = WorkerBatch(
        instance_id=batch.instance_id,
        layer_id=batch.layer_id,
        topology_version=batch.topology_version,
        hidden_states=batch.hidden_states[batch.token_indices],
        token_indices=None,
        expert_ids=batch.expert_ids,
        routing_weights=batch.routing_weights,
        distinct_expert_ids=batch.distinct_expert_ids,
    )
    hidden = torch.empty((2, 3), dtype=torch.float16)
    expert_ids = torch.empty((2, 2), dtype=torch.int32)
    routing_weights = torch.empty((2, 2), dtype=torch.float32)

    buffers.copy_input(compact, hidden, expert_ids, routing_weights)
    torch.testing.assert_close(hidden, compact.hidden_states)
    torch.testing.assert_close(expert_ids, compact.expert_ids)
    torch.testing.assert_close(routing_weights, compact.routing_weights)
    output = torch.full((2, 3), 7, dtype=torch.float16)
    assert buffers.copy_output(output, None) is output
    assert buffers.host_staging_bytes == 0

    buffers.close()
    with pytest.raises(RuntimeError, match="closed"):
        buffers.copy_output(output, None)
    run(server.close())


def test_receiver_rejects_batch_buffer_shape_mismatch() -> None:
    server = GrpcWorkerBatchReceiver(
        "127.0.0.1:0",
        batch_spec(),
        max_active_batches=1,
        max_pending_batches=1,
    )

    with pytest.raises(ValueError, match="does not match"):
        server.create_batch_buffers(
            BatchBufferConfig(
                max_batch_tokens=5,
                hidden_dim=3,
                top_k=2,
                dtype=torch.float16,
                device="cpu",
            )
        )
    run(server.close())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_receiver_reuses_pinned_cuda_batch_staging() -> None:
    server = GrpcWorkerBatchReceiver(
        "127.0.0.1:0",
        batch_spec(),
        max_active_batches=1,
        max_pending_batches=1,
    )
    buffers = server.create_batch_buffers(
        BatchBufferConfig(
            max_batch_tokens=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float16,
            device="cuda:0",
        )
    )
    batch = worker_batch()
    compact = WorkerBatch(
        instance_id=batch.instance_id,
        layer_id=batch.layer_id,
        topology_version=batch.topology_version,
        hidden_states=batch.hidden_states[batch.token_indices],
        token_indices=None,
        expert_ids=batch.expert_ids,
        routing_weights=batch.routing_weights,
        distinct_expert_ids=batch.distinct_expert_ids,
    )
    hidden = torch.empty((2, 3), dtype=torch.float16, device="cuda:0")
    expert_ids = torch.empty((2, 2), dtype=torch.int32, device="cuda:0")
    routing_weights = torch.empty((2, 2), dtype=torch.float32, device="cuda:0")
    output = torch.full((2, 3), 5, dtype=torch.float16, device="cuda:0")

    stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(stream):
        buffers.copy_input(compact, hidden, expert_ids, routing_weights)
        host_output = buffers.copy_output(output, None)
        output_pointer = host_output.data_ptr()
    stream.synchronize()
    torch.testing.assert_close(hidden.cpu(), compact.hidden_states)
    torch.testing.assert_close(host_output, torch.full((2, 3), 5, dtype=torch.float16))

    with torch.cuda.stream(stream):
        second_output = buffers.copy_output(output + 1, None)
    stream.synchronize()
    assert second_output.data_ptr() == output_pointer
    assert host_output.is_pinned()
    assert buffers.host_staging_bytes == 112
    buffers.close()
    run(server.close())


def test_receiver_hands_one_validated_batch_directly_to_execution() -> None:
    async def scenario() -> None:
        pending_changes: list[int] = []
        server, client = await start_pair(pending_changes=pending_changes)
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )

        received = await server.receive()
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

        partial_output = received.batch.hidden_states * 2
        received.release_input()
        with pytest.raises(RuntimeError, match="already been released"):
            _ = received.batch

        idle = asyncio.create_task(
            server.wait_idle(((2, 0), (2, 1), (2, 3)), monotonic_deadline=time.monotonic() + 2)
        )
        await received.complete(partial_output)
        await submission
        await idle

        torch.testing.assert_close(
            output,
            torch.tensor([[14.0, 16.0, 18.0], [2.0, 4.0, 6.0]], dtype=torch.float16),
        )
        assert server.active_count == 0
        assert server.admitted_count(2, 0) == 0
        assert pending_changes == [1, 0]
        await close_pair(server, client)

    run(scenario())


def test_unsampled_request_does_not_create_custom_spans() -> None:
    async def scenario() -> None:
        tracer = RecordingTracer(recording=False)
        server, client = await start_pair(tracer=tracer)
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )

        received = await server.receive()
        await received.complete(received.batch.hidden_states)
        await submission

        assert received.trace_context is None
        assert tracer.spans == []
        await close_pair(server, client)

    run(scenario())


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (TransportErrorCode.DEADLINE_EXCEEDED, TransportErrorCode.DEADLINE_EXCEEDED),
        (TransportErrorCode.UNAVAILABLE, TransportErrorCode.UNAVAILABLE),
    ],
)
def test_receiver_maps_native_execution_rejection(
    code: TransportErrorCode,
    expected: TransportErrorCode,
) -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        received = await server.receive()

        await received.reject(
            TransportError(code, retryable=True, diagnostic="execution did not start")
        )
        with pytest.raises(TransportError) as caught:
            await submission

        assert caught.value.code is expected
        assert server.active_count == 0
        await close_pair(server, client)

    run(scenario())


def test_receiver_delivers_structured_execution_rejection() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        received = await server.receive()
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
        await close_pair(server, client)

    run(scenario())


def test_application_pending_limit_returns_busy_and_cancellation_releases_input() -> None:
    async def scenario() -> None:
        tracer = RecordingTracer()
        server, client = await start_pair(client_in_flight=2, tracer=tracer)
        first_output = prepare_output(client)
        second_output = prepare_output(client)
        first = asyncio.create_task(
            client.execute(
                worker_batch(),
                first_output,
                monotonic_deadline=float("inf"),
            )
        )
        async with asyncio.timeout(2):
            await server._wait_pending_count(1)

        with pytest.raises(TransportError) as caught:
            await client.execute(
                worker_batch(),
                second_output,
                monotonic_deadline=float("inf"),
            )
        assert caught.value.code is TransportErrorCode.BUSY
        assert caught.value.retryable is True
        assert server.pending_count == 1
        assert server.pending_retained_bytes == 44

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        async with asyncio.timeout(2):
            await server._wait_pending_count(0)
        assert server.pending_retained_bytes == 0
        assert server.admitted_count(2, 0) == 0
        waiting = [span for span in tracer.spans if span.name == "worker.request.wait"]
        assert len(waiting) == 1
        assert waiting[0].ended is True
        assert waiting[0].attributes["expertkit.outcome"] == "cancelled"

        await close_pair(server, client)

    run(scenario())


def test_maximum_batch_accounts_only_retained_decoded_tensor_bytes() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = torch.empty((4, 3), dtype=torch.float16)
        batch = WorkerBatch(
            instance_id=7,
            layer_id=2,
            topology_version=11,
            hidden_states=torch.ones((4, 3), dtype=torch.float16),
            token_indices=None,
            expert_ids=torch.tensor(
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                dtype=torch.int32,
            ),
            routing_weights=torch.full((4, 2), 0.5, dtype=torch.float32),
            distinct_expert_ids=tuple(range(8)),
        )
        submission = asyncio.create_task(
            client.execute(batch, output, monotonic_deadline=float("inf"))
        )
        await await_with_loop_yields(server._wait_pending_count(1))

        assert server.pending_retained_bytes == (
            calculate_message_limits(batch_spec()).retained_request_tensor_bytes
        )
        received = await server.receive()
        assert server.pending_retained_bytes == 0
        await await_with_loop_yields(received.complete(received.batch.hidden_states))
        await await_with_loop_yields(submission)

        await close_pair(server, client)

    run(scenario())


def test_receiver_close_clears_pending_retained_bytes() -> None:
    async def scenario() -> None:
        tracer = RecordingTracer()
        server, client = await start_pair(tracer=tracer)
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        await await_with_loop_yields(server._wait_pending_count(1))
        assert server.pending_retained_bytes == 44

        await server.close()
        assert server.pending_count == 0
        assert server.pending_retained_bytes == 0
        waiting = [span for span in tracer.spans if span.name == "worker.request.wait"]
        assert len(waiting) == 1
        assert waiting[0].ended is True
        assert waiting[0].attributes["expertkit.outcome"] in {"cancelled", "closed"}
        result = await asyncio.gather(submission, return_exceptions=True)
        assert isinstance(result[0], TransportError)
        await client.close()

    run(scenario())


def test_expert_drain_rejects_new_calls_and_preserves_admitted_work() -> None:
    async def scenario() -> None:
        server, client = await start_pair(max_active_batches=2)
        first_output = prepare_output(client)
        rejected_output = prepare_output(client)
        unrelated_output = prepare_output(client)
        resumed_output = prepare_output(client)
        first = asyncio.create_task(
            client.execute(
                worker_batch(),
                first_output,
                monotonic_deadline=float("inf"),
            )
        )
        async with asyncio.timeout(2):
            await server._wait_pending_count(1)

        await server.begin_drain(
            ((2, 1),),
            min_topology_version=12,
            stop_all=False,
        )
        await server.begin_drain(
            ((2, 1),),
            min_topology_version=13,
            stop_all=False,
        )
        idle = asyncio.create_task(
            server.wait_idle(
                ((2, 1),),
                monotonic_deadline=time.monotonic() + 2,
            )
        )
        await asyncio.sleep(0)
        assert idle.done() is False

        with pytest.raises(TransportError) as caught:
            await await_with_loop_yields(
                client.execute(
                    worker_batch(),
                    rejected_output,
                    monotonic_deadline=float("inf"),
                )
            )
        assert caught.value.code is TransportErrorCode.DRAINING
        assert caught.value.retryable is True
        assert caught.value.min_topology_version == 13

        admitted = await server.receive()
        await await_with_loop_yields(admitted.complete(admitted.batch.hidden_states))
        await await_with_loop_yields(first)
        await idle
        assert server.admitted_count(2, 1) == 0

        unrelated_batch = replace(
            worker_batch(),
            expert_ids=torch.tensor([[2, -1], [2, -1]], dtype=torch.int32),
            routing_weights=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            distinct_expert_ids=(2,),
        )
        unrelated = asyncio.create_task(
            client.execute(
                unrelated_batch,
                unrelated_output,
                monotonic_deadline=float("inf"),
            )
        )
        await await_with_loop_yields(server._wait_pending_count(1))
        unrelated_received = await server.receive()
        await await_with_loop_yields(
            unrelated_received.complete(unrelated_received.batch.hidden_states)
        )
        await await_with_loop_yields(unrelated)

        await server.clear_drains(((2, 1),))
        await server.clear_drains(((2, 1),))
        resumed = asyncio.create_task(
            client.execute(
                worker_batch(),
                resumed_output,
                monotonic_deadline=float("inf"),
            )
        )
        await await_with_loop_yields(server._wait_pending_count(1))
        accepted = await server.receive()
        await await_with_loop_yields(accepted.complete(accepted.batch.hidden_states))
        await await_with_loop_yields(resumed)

        await await_with_loop_yields(close_pair(server, client))

    run(scenario())


def test_whole_worker_drain_rejects_every_new_batch() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        unrelated = replace(
            worker_batch(),
            expert_ids=torch.tensor([[2, -1], [2, -1]], dtype=torch.int32),
            routing_weights=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            distinct_expert_ids=(2,),
        )
        await server.begin_drain((), min_topology_version=0, stop_all=True)

        with pytest.raises(TransportError) as caught:
            await await_with_loop_yields(
                client.execute(
                    unrelated,
                    output,
                    monotonic_deadline=float("inf"),
                )
            )

        assert caught.value.code is TransportErrorCode.DRAINING
        assert caught.value.min_topology_version == 0
        assert server.pending_count == 0
        assert server.pending_retained_bytes == 0
        await await_with_loop_yields(close_pair(server, client))

    run(scenario())


def test_whole_worker_idle_wait_includes_waiting_and_active_batches() -> None:
    async def scenario() -> None:
        server, client = await start_pair(max_active_batches=2)
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        async with asyncio.timeout(2):
            await server._wait_pending_count(1)

        idle = asyncio.create_task(server.wait_idle(None, monotonic_deadline=time.monotonic() + 2))
        await asyncio.sleep(0)
        assert idle.done() is False

        received = await server.receive()
        await asyncio.sleep(0)
        assert idle.done() is False
        await await_with_loop_yields(received.complete(received.batch.hidden_states))
        await await_with_loop_yields(submission)
        await idle

        await await_with_loop_yields(close_pair(server, client))

    run(scenario())


def test_outer_grpc_concurrency_limit_bounds_active_plus_pending_calls() -> None:
    async def scenario() -> None:
        server, client = await start_pair(client_in_flight=3)
        outputs = [prepare_output(client) for _ in range(3)]
        first = asyncio.create_task(
            client.execute(
                worker_batch(),
                outputs[0],
                monotonic_deadline=float("inf"),
            )
        )
        active = await server.receive()
        others = [
            asyncio.create_task(
                client.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )
            )
            for output in outputs[1:]
        ]
        second = await server.receive()

        await active.complete(active.batch.hidden_states)
        await second.complete(second.batch.hidden_states)
        results = await asyncio.gather(first, *others, return_exceptions=True)

        failures = [result for result in results if isinstance(result, TransportError)]
        assert len(failures) == 1
        assert failures[0].code is TransportErrorCode.BUSY
        assert server.active_count == 0
        await close_pair(server, client)

    run(scenario())


def test_cancellation_after_take_discards_response_but_releases_expert_use() -> None:
    async def scenario() -> None:
        server, client = await start_pair()
        output = prepare_output(client)
        submission = asyncio.create_task(
            client.execute(
                worker_batch(),
                output,
                monotonic_deadline=float("inf"),
            )
        )
        received = await server.receive()
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
        await close_pair(server, client)

    run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"",
    ],
)
def test_receiver_rejects_malformed_wire_requests_with_native_status(payload: bytes) -> None:
    async def scenario() -> None:
        server = GrpcWorkerBatchReceiver(
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
        assert server.pending_retained_bytes == 0
        await channel.close()
        await server.close()

    run(scenario())


def test_receiver_close_wakes_execution_waiting_on_take() -> None:
    async def scenario() -> None:
        server = GrpcWorkerBatchReceiver(
            "127.0.0.1:0",
            batch_spec(),
            max_active_batches=1,
            max_pending_batches=1,
        )
        await server.start()
        waiting = asyncio.create_task(server.receive())
        await server.close()
        await server.close()

        with pytest.raises(ReceiverClosed):
            await waiting

    run(scenario())


def test_grpc_receiver_does_not_expose_shared_memory_methods() -> None:
    async def scenario() -> None:
        server = GrpcWorkerBatchReceiver(
            "127.0.0.1:0",
            batch_spec(),
            max_active_batches=1,
            max_pending_batches=1,
        )
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
        open_shared_memory = channel.unary_unary(
            "/ek.worker.v2.ComputationService/OpenSharedMemory",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await open_shared_memory(b"")
            assert caught.value.code() is grpc.StatusCode.UNIMPLEMENTED
        finally:
            await channel.close()
            await server.close()

    run(scenario())
