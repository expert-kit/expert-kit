"""Worker receiver for fixed shared-memory Tensor slots."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from functools import partial
from pathlib import Path
from typing import Any

import grpc
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import (
    TransportError,
    TransportErrorCode,
    TransportProtocolError,
)
from expertkit_transport.tracing import (
    TraceContext,
    Tracer,
    TraceSpan,
    incoming_trace_context,
)
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerBatchReceiver,
    WorkerEndpointConfig,
)
from expertkit_transport.transports.queue import ReceiverQueue
from expertkit_transport.transports.shm.codec import (
    SHM_CONTROL_MESSAGE_BYTES,
    decode_close_request,
    decode_execute_request,
    decode_open_request,
    encode_close_response,
    encode_execute_error,
    encode_execute_success,
    encode_open_response,
)
from expertkit_transport.transports.shm.session import (
    SharedMemorySlotBusy,
    WorkerSharedMemorySession,
)
from expertkit_transport.transports.shm.worker_buffers import ShmWorkerBatchBuffers

_OPEN_METHOD_NAME = "OpenSharedMemory"
_EXECUTE_METHOD_NAME = "ExecuteSharedMemory"
_CLOSE_METHOD_NAME = "CloseSharedMemory"
_SERVICE_NAME = "ek.worker.v2.ComputationService"
_MAX_SHARED_MEMORY_SESSIONS = 64
_UINT64_MAX = (1 << 64) - 1
_NATIVE_ERROR_STATUS = {
    TransportErrorCode.DEADLINE_EXCEEDED: grpc.StatusCode.DEADLINE_EXCEEDED,
    TransportErrorCode.CANCELLED: grpc.StatusCode.CANCELLED,
    TransportErrorCode.UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    TransportErrorCode.PROTOCOL: grpc.StatusCode.INVALID_ARGUMENT,
}


def _identity(payload: bytes) -> bytes:
    return payload


def _batch_trace_attributes(batch: WorkerBatch) -> dict[str, str | int]:
    return {
        "expertkit.instance_id": batch.instance_id,
        "expertkit.layer_id": batch.layer_id,
        "expertkit.topology_version": batch.topology_version,
        "expertkit.token_count": batch.token_count,
        "expertkit.assignment_count": batch.token_count * batch.top_k,
    }


class _NativeCallError(RuntimeError):
    """Carry a standard gRPC status from execution back to the RPC handler."""

    def __init__(self, status: grpc.StatusCode, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.status = status
        self.diagnostic = diagnostic


class _ShmReceivedBatch(ReceivedBatch):
    """Retain one claimed shared-memory slot until its response is complete."""

    def __init__(
        self,
        owner: ShmWorkerBatchReceiver,
        batch: WorkerBatch,
        monotonic_deadline: float,
        trace_context: TraceContext | None,
        *,
        output_destination: torch.Tensor,
        session: WorkerSharedMemorySession,
        slot_index: int,
        generation: int,
    ) -> None:
        self._owner = owner
        self._batch: WorkerBatch | None = batch
        self.token_count = batch.token_count
        self._deadline = monotonic_deadline
        self._trace_context = trace_context
        self._output_destination = output_destination
        self.session = session
        self.slot_index = slot_index
        self.generation = generation
        self._wait_span: TraceSpan | None = None
        self._cancelled = False
        self._cancelled_event = asyncio.Event()
        self.response: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

    @property
    def transport_name(self) -> str:
        return "shm"

    @property
    def trace_context(self) -> TraceContext | None:
        return self._trace_context

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("received SHM input has already been released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def output_destination(self) -> torch.Tensor:
        return self._output_destination

    def release_input(self) -> None:
        """Drop claimed input views after the execution slot has copied them."""

        self._owner._require_active(self)
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        """Publish the completed slot generation to the Frontend."""

        await self._owner._complete_success(self, partial_output)

    async def reject(self, error: TransportError) -> None:
        """Return a structured computation rejection."""

        await self._owner._complete_error(self, error)

    def start_wait_span(self, tracer: Tracer) -> None:
        """Start queue timing after successful admission."""

        if self._wait_span is not None:
            raise RuntimeError("SHM waiting span is already active")
        self._wait_span = tracer.start_span(
            "worker.queue_wait",
            context=self._trace_context,
            attributes=_batch_trace_attributes(self.batch),
        )

    def finish_wait_span(self, outcome: str) -> None:
        """Finish queue timing on receive, cancellation, or receiver close."""

        span = self._wait_span
        if span is None:
            return
        self._wait_span = None
        span.set_attribute("expertkit.outcome", outcome)
        span.end()


class ShmWorkerBatchReceiver(WorkerBatchReceiver):
    """Receive SHM slot notifications and own Worker-side session mappings."""

    def __init__(
        self,
        listen: str,
        endpoint_config: WorkerEndpointConfig,
        *,
        max_active_batches: int,
        max_pending_batches: int,
        cpu_workers: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        interceptors: Sequence[grpc.aio.ServerInterceptor] = (),
        tracer: Tracer | None = None,
        on_rejection: Callable[[str], None] | None = None,
        on_pending_changed: Callable[[int], None] | None = None,
        shared_memory_dir: Path = Path("/dev/shm"),
    ) -> None:
        if not listen:
            raise ValueError("listen must not be empty")
        for name, value in (
            ("max_active_batches", max_active_batches),
            ("max_pending_batches", max_pending_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        resolved_cpu_workers = cpu_workers if cpu_workers is not None else max_active_batches
        if (
            isinstance(resolved_cpu_workers, bool)
            or not isinstance(resolved_cpu_workers, int)
            or resolved_cpu_workers <= 0
        ):
            raise ValueError("cpu_workers must be a positive integer")

        self._listen = listen
        self._spec = endpoint_config
        self._shared_memory_dir = shared_memory_dir
        self._slot_count = max_active_batches + max_pending_batches
        self._queue = ReceiverQueue(
            max_pending_batches=max_pending_batches,
            max_retained_bytes=0,
            clock=clock,
            on_pending_changed=on_pending_changed,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=resolved_cpu_workers,
            thread_name_prefix="expertkit-shm-receiver",
        )
        self._clock = clock
        self._interceptors = tuple(interceptors)
        self._tracer = tracer
        self._on_rejection = on_rejection or (lambda _reason: None)
        self._server: grpc.aio.Server | None = None
        self._start_lock = asyncio.Lock()
        self._sessions: dict[str, WorkerSharedMemorySession] = {}
        self._session_lock = asyncio.Lock()
        self._execution_device: torch.device | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None

    @property
    def bound_port(self) -> int:
        """Return the bound notification RPC port after startup."""

        if self._bound_port is None:
            raise RuntimeError("SHM receiver has not been started")
        return self._bound_port

    @property
    def pending_count(self) -> int:
        """Return the number of admitted slot notifications waiting for execution."""

        return self._queue.pending_count

    @property
    def pending_retained_bytes(self) -> int:
        """Return zero because waiting Tensor data remains in shared memory."""

        return self._queue.retained_bytes

    @property
    def active_count(self) -> int:
        """Return batches taken by Worker execution and not yet completed."""

        return self._queue.active_count

    async def start(self) -> None:
        """Start only the SHM session and notification RPC methods."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("SHM receiver is closing")
            if self._server is not None:
                return
            options = (
                ("grpc.max_receive_message_length", SHM_CONTROL_MESSAGE_BYTES),
                ("grpc.max_send_message_length", SHM_CONTROL_MESSAGE_BYTES),
            )
            server = grpc.aio.server(
                options=options,
                maximum_concurrent_rpcs=(self._slot_count + _MAX_SHARED_MEMORY_SESSIONS),
                interceptors=self._interceptors,
            )
            service = grpc.method_handlers_generic_handler(
                _SERVICE_NAME,
                {
                    _OPEN_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
                        self._open_shared_memory,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    _EXECUTE_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
                        self._execute_shared_memory,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    _CLOSE_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
                        self._close_shared_memory,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                },
            )
            server.add_generic_rpc_handlers((service,))
            port = server.add_insecure_port(self._listen)
            if port == 0:
                raise RuntimeError("SHM receiver could not bind its RPC listen address")
            await server.start()
            self._server = server
            self._bound_port = port

    async def receive(self) -> ReceivedBatch:
        """Move one waiting slot directly into Worker execution ownership."""

        item = await self._queue.take()
        assert isinstance(item, _ShmReceivedBatch)
        item.finish_wait_span("active")
        return item

    def create_batch_buffers(self, spec: BatchBufferConfig) -> WorkerBatchBuffers:
        """Create SHM copy behavior without allocating private Host staging."""

        expected = (
            self._spec.max_batch_tokens,
            self._spec.hidden_dim,
            self._spec.top_k,
            self._spec.dtype,
        )
        actual = (spec.max_batch_tokens, spec.hidden_dim, spec.top_k, spec.dtype)
        if actual != expected:
            raise ValueError("Worker buffer shape does not match the SHM endpoint")
        if self._execution_device is None:
            self._execution_device = spec.device
        elif self._execution_device != spec.device:
            raise ValueError("all Worker execution slots must use the same device")
        return ShmWorkerBatchBuffers(spec)

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        """Reject new matching requests while preserving admitted work."""

        await self._queue.begin_drain(
            experts,
            min_topology_version=min_topology_version,
            stop_all=stop_all,
        )

    async def clear_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        """Clear per-expert gates after later assignments become ready."""

        await self._queue.clear_expert_drains(experts)

    async def wait_idle(
        self,
        experts: Iterable[tuple[int, int]] | None,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Wait for selected expert use, or all work when experts is `None`."""

        if experts is None:
            await self._queue.wait_all_idle(monotonic_deadline=monotonic_deadline)
        else:
            await self._queue.wait_experts_idle(
                experts,
                monotonic_deadline=monotonic_deadline,
            )

    async def close(self) -> None:
        """Stop notifications and close all idle shared-memory sessions."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    def admitted_count(self, layer_id: int, expert_id: int) -> int:
        """Return waiting plus active batches that name one expert."""

        return self._queue.admitted_count(layer_id, expert_id)

    async def _open_shared_memory(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        try:
            description = decode_open_request(
                payload,
                self._spec,
                expected_slot_count=self._slot_count,
            )
        except TransportProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        device = self._execution_device
        if device is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Worker execution slots are not initialized",
            )
            raise AssertionError("context.abort must terminate the handler")

        async with self._session_lock:
            if description.session_id in self._sessions:
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    "shared-memory session already exists",
                )
                raise AssertionError("context.abort must terminate the handler")
            if len(self._sessions) >= _MAX_SHARED_MEMORY_SESSIONS:
                await context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "Worker has too many shared-memory sessions",
                )
                raise AssertionError("context.abort must terminate the handler")
            try:
                session = await self._run_cpu(
                    partial(
                        WorkerSharedMemorySession,
                        description,
                        spec=self._spec,
                        device=device,
                        directory=self._shared_memory_dir,
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"cannot open shared-memory segment: {error}",
                )
                raise AssertionError("context.abort must terminate the handler") from error
            self._sessions[description.session_id] = session
        return encode_open_response()

    async def _execute_shared_memory(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            self._record_rejection(TransportErrorCode.UNAVAILABLE.value)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        trace_context = incoming_trace_context(self._tracer, context.invocation_metadata())
        trace_enabled = trace_context is not None
        try:
            with self._trace_span(
                "worker.request_decode",
                context=trace_context,
                attributes={"expertkit.request_bytes": len(payload)},
                enabled=trace_enabled,
            ) as span:
                request = decode_execute_request(payload, self._spec)
                session = self._sessions.get(request.session_id)
                if session is None:
                    await context.abort(
                        grpc.StatusCode.NOT_FOUND,
                        "shared-memory session is not registered",
                    )
                    raise AssertionError("context.abort must terminate the handler")
                claimed = session.claim(request)
                if span is not None:
                    for key, value in _batch_trace_attributes(claimed.batch).items():
                        span.set_attribute(key, value)
                    tensor_bytes = request.token_count * (
                        2 * self._spec.hidden_dim * self._spec.activation_element_bytes
                        + 2 * self._spec.top_k * 4
                    )
                    span.set_attribute("expertkit.shared_tensor_bytes", tensor_bytes)
        except SharedMemorySlotBusy as error:
            self._record_rejection(TransportErrorCode.BUSY.value)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        except TransportProtocolError as error:
            self._record_rejection(TransportErrorCode.PROTOCOL.value)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error

        deadline = (
            math.inf
            if request.timeout_micros == _UINT64_MAX
            else self._clock() + request.timeout_micros / 1_000_000
        )
        item = _ShmReceivedBatch(
            self,
            claimed.batch,
            deadline,
            trace_context,
            output_destination=claimed.output_destination,
            session=session,
            slot_index=claimed.slot_index,
            generation=claimed.generation,
        )
        try:
            rejection = await self._queue.admit(item, retained_bytes=0)
        except BaseException:
            session.release(claimed.slot_index, claimed.generation)
            raise
        if rejection is not None:
            session.release(claimed.slot_index, claimed.generation)
            self._record_rejection(rejection.code.value)
            return encode_execute_error(rejection, self._spec)
        if self._tracer is not None and item.trace_context is not None:
            item.start_wait_span(self._tracer)

        try:
            return await asyncio.shield(item.response)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._cancel(item))
            with suppress(BaseException):
                await asyncio.shield(cleanup)
            raise
        except _NativeCallError as error:
            await context.abort(error.status, error.diagnostic)
            raise AssertionError("context.abort must terminate the handler") from error
        except Exception as error:
            await context.abort(grpc.StatusCode.INTERNAL, "failed to finish shared-memory response")
            raise AssertionError("context.abort must terminate the handler") from error

    async def _close_shared_memory(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        try:
            session_id = decode_close_request(payload)
        except TransportProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        async with self._session_lock:
            session = self._sessions.get(session_id)
            if session is None:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    "shared-memory session is not registered",
                )
                raise AssertionError("context.abort must terminate the handler")
            if session.active_count:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "shared-memory session still has active slots",
                )
                raise AssertionError("context.abort must terminate the handler")
            self._sessions.pop(session_id)
            await self._run_cpu(session.close)
        return encode_close_response()

    async def _cancel(self, item: _ShmReceivedBatch) -> None:
        item._cancelled = True
        item._cancelled_event.set()
        if await self._queue.cancel_waiting(item):
            item.finish_wait_span("cancelled")
            self._release_slot(item)

    async def _wait_pending_count(self, expected: int) -> None:
        await self._queue.wait_for_pending_count(expected)

    async def _wait_cancelled(self, item: _ShmReceivedBatch) -> None:
        await item._cancelled_event.wait()

    async def _complete_success(
        self,
        item: _ShmReceivedBatch,
        partial_output: torch.Tensor,
    ) -> None:
        self._require_active(item)
        try:
            if not item.cancelled:
                if partial_output.ndim != 2 or partial_output.shape[0] != item.token_count:
                    raise ValueError("partial output token count does not match the received batch")
                if partial_output.data_ptr() != item.output_destination.data_ptr():
                    raise ValueError("SHM response did not use its claimed output destination")
                with self._trace_span(
                    "worker.response_encode",
                    enabled=item.trace_context is not None,
                ):
                    payload = encode_execute_success(item.generation)
                if not item.cancelled and not item.response.done():
                    item.response.set_result(payload)
        except BaseException as error:
            if not item.cancelled and not item.response.done():
                item.response.set_exception(error)
            raise
        finally:
            await self._finish_active(item)

    async def _complete_error(
        self,
        item: _ShmReceivedBatch,
        error: TransportError,
    ) -> None:
        self._require_active(item)
        try:
            if not item.cancelled:
                status = _NATIVE_ERROR_STATUS.get(error.code)
                if status is None:
                    with self._trace_span(
                        "worker.response_encode",
                        enabled=item.trace_context is not None,
                    ):
                        payload = encode_execute_error(error, self._spec)
                    if not item.cancelled and not item.response.done():
                        item.response.set_result(payload)
                elif not item.response.done():
                    item.response.set_exception(
                        _NativeCallError(status, error.diagnostic or error.code.value)
                    )
        except BaseException as cause:
            if not item.cancelled and not item.response.done():
                item.response.set_exception(cause)
            raise
        finally:
            await self._finish_active(item)

    async def _finish_active(self, item: _ShmReceivedBatch) -> None:
        item._batch = None
        cleanup = asyncio.create_task(self._release_active(item))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _release_active(self, item: _ShmReceivedBatch) -> None:
        self._release_slot(item)
        await self._queue.finish(item)

    @staticmethod
    def _release_slot(item: _ShmReceivedBatch) -> None:
        item.session.release(item.slot_index, item.generation)

    def _require_active(self, item: _ShmReceivedBatch) -> None:
        self._queue.require_active(item)

    def _trace_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: dict[str, str | int] | None = None,
        enabled: bool = True,
    ) -> Any:
        if self._tracer is None or not enabled:
            return nullcontext(None)
        return self._tracer.start_as_current_span(
            name,
            context=context,
            attributes=attributes,
        )

    def _record_rejection(self, reason: str) -> None:
        with suppress(Exception):
            self._on_rejection(reason)

    async def _close(self) -> None:
        self._closing = True
        if self._server is not None:
            await self._server.stop(None)
        waiting = await self._queue.begin_close()
        for received in waiting:
            assert isinstance(received, _ShmReceivedBatch)
            received._cancelled = True
            received._cancelled_event.set()
            received.finish_wait_span("closed")
            self._release_slot(received)
        await self._queue.wait_active_empty()
        async with self._session_lock:
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                await self._run_cpu(session.close)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._executor.shutdown, wait=True, cancel_futures=True),
        )

    async def _run_cpu(self, function: Callable[..., Any], *args: object) -> Any:
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(self._executor, function, *args)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await work
            raise
