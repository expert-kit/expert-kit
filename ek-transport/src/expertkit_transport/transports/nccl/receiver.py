"""Worker NCCL receiver with gRPC admission and mandatory P2P matching."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext, suppress
from typing import Any

import grpc
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import (
    TransportError,
    TransportErrorCode,
    TransportProtocolError,
)
from expertkit_transport.tracing import TraceContext, Tracer, TraceSpan
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerBatchReceiver,
    WorkerEndpointConfig,
)
from expertkit_transport.transports.nccl.codec import (
    NCCL_CONTROL_MESSAGE_BYTES,
    NcclExecute,
    decode_execute_request,
    decode_hello_request,
    encode_admitted,
    encode_error,
    encode_hello_response,
    encode_success,
)
from expertkit_transport.transports.nccl.runtime import (
    NcclRuntimeProtocol,
    NcclWorkerExchangeProtocol,
)
from expertkit_transport.transports.nccl.worker_buffers import (
    NcclReceiveBufferPool,
    NcclReceiveBuffers,
    NcclWorkerBatchBuffers,
)
from expertkit_transport.transports.queue import ReceiverQueue

_HELLO_METHOD_NAME = "Hello"
_EXECUTE_METHOD_NAME = "Execute"
_SERVICE_NAME = "ek.worker.v2.NcclComputationService"
_UINT64_MAX = (1 << 64) - 1


def _identity(payload: bytes) -> bytes:
    return payload


def _batch_trace_attributes(batch: WorkerBatch) -> dict[str, str | int]:
    return {
        "expertkit.instance_id": batch.instance_id,
        "expertkit.layer_id": batch.layer_id,
        "expertkit.topology_version": batch.topology_version,
        "expertkit.token_count": batch.token_count,
        "expertkit.assignment_count": batch.token_count * batch.top_k,
        "expertkit.transport": "nccl",
    }


class _InputReceiveFailed(RuntimeError):
    """Tell ``receive`` to skip an admitted item whose NCCL receive failed."""


class _NcclReceivedBatch(ReceivedBatch):
    def __init__(
        self,
        owner: NcclWorkerBatchReceiver,
        request: NcclExecute,
        batch: WorkerBatch,
        buffers: NcclReceiveBuffers,
        monotonic_deadline: float,
        trace_context: TraceContext | None,
    ) -> None:
        self._owner = owner
        self.request = request
        self.buffers = buffers
        self._batch: WorkerBatch | None = batch
        self._deadline = monotonic_deadline
        self._trace_context = trace_context
        self._wait_span: TraceSpan | None = None
        self._cancelled = False
        self._input_ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._response: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self.exchange: NcclWorkerExchangeProtocol | None = None
        self.control_peer_lock: asyncio.Lock | None = None
        self.released = False

    @property
    def trace_context(self) -> TraceContext | None:
        return self._trace_context

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("received NCCL input has already been released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def output_destination(self) -> torch.Tensor:
        return self.buffers.partial_output[: self.request.token_count]

    def release_input(self) -> None:
        self._owner._require_active(self)
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        await self._owner._complete_success(self, partial_output)

    async def reject(self, error: TransportError) -> None:
        await self._owner._complete_error(self, error)

    def start_wait_span(self, tracer: Tracer) -> None:
        if self._wait_span is not None:
            raise RuntimeError("NCCL waiting span is already active")
        self._wait_span = tracer.start_span(
            "worker.request.wait",
            context=self._trace_context,
            attributes=_batch_trace_attributes(self.batch),
        )

    def finish_wait_span(self, outcome: str) -> None:
        span = self._wait_span
        if span is None:
            return
        self._wait_span = None
        span.set_attribute("expertkit.outcome", outcome)
        span.end()


class NcclWorkerBatchReceiver(WorkerBatchReceiver):
    """Admit metadata over gRPC and receive CUDA tensors over shared NCCL."""

    def __init__(
        self,
        control_listen: str,
        endpoint_config: WorkerEndpointConfig,
        *,
        runtime: NcclRuntimeProtocol,
        max_active_batches: int,
        max_pending_batches: int,
        owns_runtime: bool = True,
        clock: Callable[[], float] = time.monotonic,
        interceptors: Sequence[grpc.aio.ServerInterceptor] = (),
        tracer: Tracer | None = None,
        on_rejection: Callable[[str], None] | None = None,
        on_pending_changed: Callable[[int], None] | None = None,
    ) -> None:
        if not control_listen:
            raise ValueError("control_listen must not be empty")
        for name, value in (
            ("max_active_batches", max_active_batches),
            ("max_pending_batches", max_pending_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(owns_runtime, bool):
            raise ValueError("owns_runtime must be a Boolean")

        self._listen = control_listen
        self._spec = endpoint_config
        self._runtime = runtime
        self._owns_runtime = owns_runtime
        self._slot_count = max_active_batches + max_pending_batches
        self._pool = NcclReceiveBufferPool(
            endpoint_config,
            device=runtime.device,
            capacity=self._slot_count,
        )
        self._queue = ReceiverQueue(
            max_pending_batches=max_pending_batches,
            max_retained_bytes=0,
            clock=clock,
            on_pending_changed=on_pending_changed,
        )
        self._clock = clock
        self._interceptors = tuple(interceptors)
        self._tracer = tracer
        self._on_rejection = on_rejection or (lambda _reason: None)
        self._server: grpc.aio.Server | None = None
        self._start_lock = asyncio.Lock()
        self._execution_device: torch.device | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None
        self._drive_tasks: set[asyncio.Task[bytes]] = set()
        self._control_peer_locks: dict[int, asyncio.Lock] = {}

    @property
    def bound_port(self) -> int:
        if self._bound_port is None:
            raise RuntimeError("NCCL receiver has not been started")
        return self._bound_port

    @property
    def pending_count(self) -> int:
        return self._queue.pending_count

    @property
    def pending_retained_bytes(self) -> int:
        return 0

    @property
    def active_count(self) -> int:
        return self._queue.active_count

    async def start(self) -> None:
        """Start control admission without waiting for every static NCCL rank."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("NCCL receiver is closing")
            if self._server is not None:
                return
            options = (
                ("grpc.max_receive_message_length", NCCL_CONTROL_MESSAGE_BYTES),
                ("grpc.max_send_message_length", NCCL_CONTROL_MESSAGE_BYTES),
            )
            server = grpc.aio.server(
                options=options,
                maximum_concurrent_rpcs=(self._slot_count + 8),
                interceptors=self._interceptors,
            )
            service = grpc.method_handlers_generic_handler(
                _SERVICE_NAME,
                {
                    _HELLO_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
                        self._hello,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    _EXECUTE_METHOD_NAME: grpc.unary_stream_rpc_method_handler(
                        self._execute,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                },
            )
            server.add_generic_rpc_handlers((service,))
            port = server.add_insecure_port(self._listen)
            if port == 0:
                raise RuntimeError("NCCL receiver could not bind its control listen address")
            await server.start()
            self._server = server
            self._bound_port = port

    async def receive(self) -> ReceivedBatch:
        """Take one item only after all three NCCL input receives have matched."""

        while True:
            item = await self._queue.take()
            assert isinstance(item, _NcclReceivedBatch)
            item.finish_wait_span("active")
            try:
                await asyncio.shield(item._input_ready)
            except _InputReceiveFailed:
                await self._finish_failed_active(item)
                continue
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(self._cancel_active_take(item))
                with suppress(BaseException):
                    await asyncio.shield(cleanup)
                raise
            return item

    def create_batch_buffers(self, spec: BatchBufferConfig) -> WorkerBatchBuffers:
        expected = (
            self._spec.max_batch_tokens,
            self._spec.hidden_dim,
            self._spec.top_k,
            self._spec.dtype,
            self._runtime.device,
        )
        actual = (
            spec.max_batch_tokens,
            spec.hidden_dim,
            spec.top_k,
            spec.dtype,
            spec.device,
        )
        if actual != expected:
            raise ValueError("Worker buffer shape/device does not match the NCCL endpoint")
        if self._execution_device is None:
            self._execution_device = spec.device
        elif self._execution_device != spec.device:
            raise ValueError("all Worker execution slots must use the same device")
        return NcclWorkerBatchBuffers(
            spec,
            experts_per_layer=self._spec.experts_per_layer,
        )

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        await self._queue.begin_drain(
            experts,
            min_topology_version=min_topology_version,
            stop_all=stop_all,
        )

    async def clear_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        await self._queue.clear_expert_drains(experts)

    async def wait_idle(
        self,
        experts: Iterable[tuple[int, int]] | None,
        *,
        monotonic_deadline: float,
    ) -> None:
        if experts is None:
            await self._queue.wait_all_idle(monotonic_deadline=monotonic_deadline)
        else:
            await self._queue.wait_experts_idle(
                experts,
                monotonic_deadline=monotonic_deadline,
            )

    async def close(self) -> None:
        if self._close_task is None:
            # Stop admission before cleanup gets its first event-loop turn.
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    def admitted_count(self, layer_id: int, expert_id: int) -> int:
        return self._queue.admitted_count(layer_id, expert_id)

    async def _hello(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        try:
            decode_hello_request(payload, self._runtime, self._spec)
        except TransportProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        # Delay the finite static rendezvous until a validated Frontend has
        # discovered this registered Worker.  Starting it during Worker boot
        # would consume the timeout while model loading/publication is ongoing.
        await self._runtime.start()
        return encode_hello_response(self._runtime, self._spec)

    async def _execute(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        if self._closing:
            self._record_rejection(TransportErrorCode.UNAVAILABLE.value)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        trace_enabled = self._tracer is not None and self._tracer.current_span_is_recording()
        trace_context = self._tracer.capture_context() if trace_enabled else None
        try:
            request = decode_execute_request(payload, self._runtime, self._spec)
        except TransportProtocolError as error:
            self._record_rejection(TransportErrorCode.PROTOCOL.value)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        del payload

        context_remaining = context.time_remaining()
        request_remaining = (
            math.inf
            if request.timeout_micros == _UINT64_MAX
            else request.timeout_micros / 1_000_000
        )
        remaining = (
            request_remaining
            if context_remaining is None
            else min(request_remaining, max(0.0, context_remaining))
        )
        deadline = math.inf if math.isinf(remaining) else self._clock() + remaining
        try:
            await self._runtime.wait_ready(monotonic_deadline=deadline)
        except TransportError as error:
            self._record_rejection(error.code.value)
            yield encode_error(request.sequence, error)
            return

        control_peer_lock = self._control_peer_locks.setdefault(
            request.client_rank,
            asyncio.Lock(),
        )
        if control_peer_lock.locked():
            error = TransportError(
                TransportErrorCode.BUSY,
                retryable=True,
                diagnostic="the NCCL peer already has an admitted operation",
            )
            self._record_rejection(error.code.value)
            yield encode_error(request.sequence, error)
            return
        await control_peer_lock.acquire()

        buffers = self._pool.take()
        if buffers is None:
            control_peer_lock.release()
            error = TransportError(
                TransportErrorCode.BUSY,
                retryable=True,
                diagnostic="Worker NCCL receive slots are full",
            )
            self._record_rejection(error.code.value)
            yield encode_error(request.sequence, error)
            return
        token_count = request.token_count
        batch = WorkerBatch(
            instance_id=self._spec.instance_id,
            layer_id=request.layer_id,
            topology_version=request.topology_version,
            hidden_states=buffers.hidden_states[:token_count],
            token_indices=None,
            expert_ids=buffers.expert_ids[:token_count],
            routing_weights=buffers.routing_weights[:token_count],
            distinct_expert_ids=request.distinct_expert_ids,
        )
        item = _NcclReceivedBatch(
            self,
            request,
            batch,
            buffers,
            deadline,
            trace_context,
        )
        item.control_peer_lock = control_peer_lock
        try:
            rejection = await self._queue.admit(item, retained_bytes=0)
        except BaseException:
            self._release_buffers(item)
            raise
        if rejection is not None:
            self._release_buffers(item)
            self._record_rejection(rejection.code.value)
            yield encode_error(request.sequence, rejection)
            return
        if self._tracer is not None and item.trace_context is not None:
            item.start_wait_span(self._tracer)

        # Nothing in the data plane is posted before this ACK.  Once yielded,
        # the remainder is an uninterruptible four-operation matching phase.
        yield encode_admitted(request.sequence)
        drive = asyncio.create_task(
            self._drive_exchange(item),
            name=f"nccl-worker-sequence-{request.sequence}",
        )
        self._drive_tasks.add(drive)
        drive.add_done_callback(self._drive_tasks.discard)
        try:
            terminal = await asyncio.shield(drive)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._cancel(item))
            with suppress(BaseException):
                await asyncio.shield(cleanup)
            terminal = await asyncio.shield(drive)
            return
        yield terminal

    async def _drive_exchange(self, item: _NcclReceivedBatch) -> bytes:
        request = item.request
        try:
            exchange = await self._runtime.receive_inputs(
                request.client_rank,
                item.buffers.hidden_states[: request.token_count],
                item.buffers.expert_ids[: request.token_count],
                item.buffers.routing_weights[: request.token_count],
                item.buffers.partial_output[: request.token_count],
                # The ACK has committed this peer to all four untagged ops.
                monotonic_deadline=math.inf,
            )
            item.exchange = exchange
            if not item._input_ready.done():
                item._input_ready.set_result(None)
            return await asyncio.shield(item._response)
        except BaseException as error:
            if item.exchange is None:
                await self._fail_input_receive(item, error)
            raise

    async def _cancel(self, item: _NcclReceivedBatch) -> None:
        item._cancelled = True
        if await self._queue.cancel_waiting(item):
            item.finish_wait_span("cancelled")
            await self._finish_cancelled_waiting(item)

    async def _finish_cancelled_waiting(self, item: _NcclReceivedBatch) -> None:
        try:
            with suppress(_InputReceiveFailed):
                await asyncio.shield(item._input_ready)
            exchange = item.exchange
            if exchange is not None:
                await exchange.abort(monotonic_deadline=math.inf)
            if not item._response.done():
                item._response.set_result(
                    encode_error(
                        item.request.sequence,
                        TransportError(
                            TransportErrorCode.CANCELLED,
                            retryable=False,
                            diagnostic="NCCL caller cancelled before Worker execution",
                        ),
                    )
                )
        except BaseException as error:
            if not item._response.done():
                item._response.set_exception(error)
            raise
        finally:
            self._release_buffers(item)

    async def _cancel_active_take(self, item: _NcclReceivedBatch) -> None:
        """Match the data plane when a receive loop stops after queue take."""

        item._cancelled = True
        try:
            await asyncio.shield(item._input_ready)
        except _InputReceiveFailed:
            await self._finish_failed_active(item)
            return
        exchange = item.exchange
        try:
            if exchange is not None:
                await exchange.abort(monotonic_deadline=math.inf)
            if not item._response.done():
                item._response.set_result(
                    encode_error(
                        item.request.sequence,
                        TransportError(
                            TransportErrorCode.CANCELLED,
                            retryable=False,
                            diagnostic="Worker receive loop stopped before execution",
                        ),
                    )
                )
        except BaseException as error:
            if not item._response.done():
                item._response.set_exception(error)
            raise
        finally:
            item._batch = None
            await self._queue.finish(item)
            self._release_buffers(item)

    async def _complete_success(
        self,
        item: _NcclReceivedBatch,
        partial_output: torch.Tensor,
    ) -> None:
        self._require_active(item)
        exchange = item.exchange
        sent = False
        try:
            destination = item.output_destination
            if partial_output.shape != destination.shape:
                raise ValueError("partial output shape does not match the NCCL request")
            if (
                partial_output.dtype != self._spec.dtype
                or partial_output.device != self._runtime.device
            ):
                raise ValueError("partial output dtype/device does not match the NCCL endpoint")
            if partial_output.data_ptr() != destination.data_ptr():
                raise ValueError("NCCL response did not use its fixed output destination")
            if exchange is None:
                raise RuntimeError("NCCL input exchange has not completed")
            await exchange.send_output(partial_output, monotonic_deadline=math.inf)
            sent = True
            if not item._response.done():
                item._response.set_result(encode_success(item.request.sequence))
        except BaseException as error:
            if exchange is not None and not sent:
                with suppress(BaseException):
                    await exchange.abort(monotonic_deadline=math.inf)
            if not item._response.done():
                item._response.set_exception(error)
            raise
        finally:
            await self._finish_active(item)

    async def _complete_error(
        self,
        item: _NcclReceivedBatch,
        error: TransportError,
    ) -> None:
        self._require_active(item)
        try:
            exchange = item.exchange
            if exchange is None:
                raise RuntimeError("NCCL input exchange has not completed")
            # A rejected request still owes the caller's already-posted irecv.
            await exchange.abort(monotonic_deadline=math.inf)
            if not item._response.done():
                item._response.set_result(encode_error(item.request.sequence, error))
        except BaseException as cause:
            if not item._response.done():
                item._response.set_exception(cause)
            raise
        finally:
            await self._finish_active(item)

    async def _finish_active(self, item: _NcclReceivedBatch) -> None:
        item._batch = None
        cleanup = asyncio.create_task(self._release_active(item))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _release_active(self, item: _NcclReceivedBatch) -> None:
        await self._queue.finish(item)
        self._release_buffers(item)

    async def _finish_failed_active(self, item: _NcclReceivedBatch) -> None:
        await self._queue.finish(item)
        self._release_buffers(item)

    async def _fail_input_receive(
        self,
        item: _NcclReceivedBatch,
        error: BaseException,
    ) -> None:
        marker = _InputReceiveFailed(str(error))
        marker.__cause__ = error
        if not item._input_ready.done():
            item._input_ready.set_exception(marker)
        if await self._queue.cancel_waiting(item):
            item.finish_wait_span("receive_failed")
            # Consume the Future exception when no receiver can observe it.
            with suppress(BaseException):
                item._input_ready.exception()
            self._release_buffers(item)

    def _release_buffers(self, item: _NcclReceivedBatch) -> None:
        if item.released:
            return
        item.released = True
        item._batch = None
        self._pool.put(item.buffers)
        control_peer_lock = item.control_peer_lock
        item.control_peer_lock = None
        if control_peer_lock is not None:
            control_peer_lock.release()

    def _require_active(self, item: _NcclReceivedBatch) -> None:
        self._queue.require_active(item)

    def _record_rejection(self, reason: str) -> None:
        with suppress(Exception):
            self._on_rejection(reason)

    def _trace_span(
        self,
        name: str,
        *,
        attributes: dict[str, str | int] | None = None,
        enabled: bool = True,
    ) -> Any:
        if self._tracer is None or not enabled:
            return nullcontext(None)
        return self._tracer.start_as_current_span(name, attributes=attributes)

    async def _close(self) -> None:
        self._closing = True
        if self._server is not None:
            await self._server.stop(None)
        waiting = await self._queue.begin_close()
        for received in waiting:
            assert isinstance(received, _NcclReceivedBatch)
            received._cancelled = True
            received.finish_wait_span("closed")
            await self._finish_cancelled_waiting(received)
        await self._queue.wait_active_empty()
        if self._drive_tasks:
            await asyncio.gather(*tuple(self._drive_tasks), return_exceptions=True)
        self._pool.close()
        if self._owns_runtime:
            await self._runtime.close()


__all__ = ["NcclWorkerBatchReceiver"]
