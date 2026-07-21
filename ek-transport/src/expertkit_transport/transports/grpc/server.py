"""Worker-side gRPC receiver with one bounded pending area."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from functools import partial
from typing import Any

import grpc
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.tracing import TraceContext, Tracer, TraceSpan
from expertkit_transport.transports.base import (
    ReceivedWorkerBatch,
    WorkerBatchReceiver,
    WorkerPositionBuffers,
    WorkerPositionSpec,
)
from expertkit_transport.transports.grpc.codec import (
    GrpcProtocolError,
    decode_request_with_size,
    encode_error_response,
    encode_success_response,
)
from expertkit_transport.transports.grpc.spec import (
    GrpcBatchSpec,
    calculate_message_limits,
)
from expertkit_transport.transports.grpc.worker_buffers import GrpcWorkerPositionBuffers
from expertkit_transport.transports.queue import ReceiverQueue
from expertkit_transport.transports.shm.codec import (
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

_EXECUTE_METHOD_NAME = "Execute"
_OPEN_SHARED_MEMORY_METHOD_NAME = "OpenSharedMemory"
_EXECUTE_SHARED_MEMORY_METHOD_NAME = "ExecuteSharedMemory"
_CLOSE_SHARED_MEMORY_METHOD_NAME = "CloseSharedMemory"
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


class _GrpcWorkItem(ReceivedWorkerBatch):
    def __init__(
        self,
        owner: GrpcWorkerServer,
        batch: WorkerBatch,
        monotonic_deadline: float,
        retained_bytes: int,
        trace_context: TraceContext | None,
        *,
        output_destination: torch.Tensor | None = None,
        shared_session: WorkerSharedMemorySession | None = None,
        shared_slot_index: int | None = None,
        shared_generation: int | None = None,
    ) -> None:
        self._owner = owner
        self._batch: WorkerBatch | None = batch
        self.layer_id = batch.layer_id
        self.token_count = batch.token_count
        self.distinct_expert_ids = batch.distinct_expert_ids
        self._deadline = monotonic_deadline
        self.retained_bytes = retained_bytes
        self._trace_context = trace_context
        self._output_destination = output_destination
        self.shared_session = shared_session
        self.shared_slot_index = shared_slot_index
        self.shared_generation = shared_generation
        self._wait_span: TraceSpan | None = None
        self._cancelled = False
        self._cancelled_event = asyncio.Event()
        self.response: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

    @property
    def trace_context(self) -> TraceContext | None:
        return self._trace_context

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("received gRPC input has already been released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def output_destination(self) -> torch.Tensor | None:
        return self._output_destination

    def release_input(self) -> None:
        """Drop decoded protobuf Tensor views after the active-position copy."""

        self._owner._require_active(self)
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        """Serialize a success unless the caller already discarded the response."""

        await self._owner._complete_success(self, partial_output)

    async def reject(self, error: TransportError) -> None:
        """Serialize one structured computation rejection."""

        await self._owner._complete_error(self, error)

    def start_wait_span(self, tracer: Tracer) -> None:
        """Start queue timing after successful admission."""

        if self._wait_span is not None:
            raise RuntimeError("gRPC waiting span is already active")
        self._wait_span = tracer.start_span(
            "worker.request.wait",
            context=self._trace_context,
            attributes=_batch_trace_attributes(self.batch),
        )

    def finish_wait_span(self, outcome: str) -> None:
        """Finish queue timing on take, cancellation, or receiver close."""

        span = self._wait_span
        if span is None:
            return
        self._wait_span = None
        span.set_attribute("expertkit.outcome", outcome)
        span.end()


class GrpcWorkerServer(WorkerBatchReceiver):
    """Receive plaintext unary calls and expose one Transport-owned waiting area."""

    def __init__(
        self,
        listen: str,
        batch_spec: GrpcBatchSpec,
        *,
        max_active_batches: int,
        max_pending_batches: int,
        cpu_workers: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        interceptors: Sequence[grpc.aio.ServerInterceptor] = (),
        tracer: Tracer | None = None,
        on_rejection: Callable[[str], None] | None = None,
        on_pending_changed: Callable[[int], None] | None = None,
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
        self._spec = batch_spec
        self._limits = calculate_message_limits(batch_spec)
        self._maximum_concurrent_rpcs = max_active_batches + max_pending_batches
        self._queue = ReceiverQueue(
            max_pending_batches=max_pending_batches,
            max_retained_bytes=(max_pending_batches * self._limits.retained_request_tensor_bytes),
            clock=clock,
            on_pending_changed=on_pending_changed,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=resolved_cpu_workers,
            thread_name_prefix="expertkit-grpc-server",
        )
        self._clock = clock
        self._interceptors = tuple(interceptors)
        self._tracer = tracer
        self._on_rejection = on_rejection or (lambda _reason: None)
        self._server: grpc.aio.Server | None = None
        self._start_lock = asyncio.Lock()
        self._shared_sessions: dict[str, WorkerSharedMemorySession] = {}
        self._shared_session_lock = asyncio.Lock()
        self._position_device: torch.device | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None

    @property
    def bound_port(self) -> int:
        """Return the bound TCP port after startup."""

        if self._bound_port is None:
            raise RuntimeError("gRPC receiver has not been started")
        return self._bound_port

    @property
    def pending_count(self) -> int:
        """Return the number of decoded calls waiting for Worker execution."""

        return self._queue.pending_count

    @property
    def pending_retained_bytes(self) -> int:
        """Return raw Tensor bytes retained by waiting calls."""

        return self._queue.retained_bytes

    @property
    def active_count(self) -> int:
        """Return batches already taken by Worker execution."""

        return self._queue.active_count

    async def start(self) -> None:
        """Start the finite plaintext `grpc.aio` computation server."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("gRPC receiver is closing")
            if self._server is not None:
                return
            server = grpc.aio.server(
                options=self._limits.server_options,
                maximum_concurrent_rpcs=self._maximum_concurrent_rpcs,
                interceptors=self._interceptors,
            )
            method = grpc.unary_unary_rpc_method_handler(
                self._execute,
                request_deserializer=_identity,
                response_serializer=_identity,
            )
            open_shared_memory = grpc.unary_unary_rpc_method_handler(
                self._open_shared_memory,
                request_deserializer=_identity,
                response_serializer=_identity,
            )
            execute_shared_memory = grpc.unary_unary_rpc_method_handler(
                self._execute_shared_memory,
                request_deserializer=_identity,
                response_serializer=_identity,
            )
            close_shared_memory = grpc.unary_unary_rpc_method_handler(
                self._close_shared_memory,
                request_deserializer=_identity,
                response_serializer=_identity,
            )
            service = grpc.method_handlers_generic_handler(
                _SERVICE_NAME,
                {
                    _EXECUTE_METHOD_NAME: method,
                    _OPEN_SHARED_MEMORY_METHOD_NAME: open_shared_memory,
                    _EXECUTE_SHARED_MEMORY_METHOD_NAME: execute_shared_memory,
                    _CLOSE_SHARED_MEMORY_METHOD_NAME: close_shared_memory,
                },
            )
            server.add_generic_rpc_handlers((service,))
            port = server.add_insecure_port(self._listen)
            if port == 0:
                raise RuntimeError("gRPC receiver could not bind its listen address")
            await server.start()
            self._server = server
            self._bound_port = port

    async def take(self) -> ReceivedWorkerBatch:
        """Move one waiting batch directly into Worker execution ownership."""

        item = await self._queue.take()
        item.finish_wait_span("active")
        return item

    def allocate_position_buffers(self, spec: WorkerPositionSpec) -> WorkerPositionBuffers:
        """Allocate fixed gRPC staging for one Worker active position."""

        expected = (
            self._spec.max_batch_tokens,
            self._spec.hidden_dim,
            self._spec.top_k,
            self._spec.dtype,
        )
        actual = (spec.max_batch_tokens, spec.hidden_dim, spec.top_k, spec.dtype)
        if actual != expected:
            raise ValueError("Worker position shape does not match the gRPC endpoint")
        if self._position_device is None:
            self._position_device = spec.device
        elif self._position_device != spec.device:
            raise ValueError("all Worker positions must use the same device")
        return GrpcWorkerPositionBuffers(spec)

    async def close(self) -> None:
        """Stop RPC admission and release waiting calls and CPU workers."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        """Reject new matching calls while preserving already admitted work."""

        await self._queue.begin_drain(
            experts,
            min_topology_version=min_topology_version,
            stop_all=stop_all,
        )

    async def clear_expert_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        """Clear per-expert gates after later assignments become ready."""

        await self._queue.clear_expert_drains(experts)

    async def wait_experts_idle(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        monotonic_deadline: float,
    ) -> None:
        """Wait until no waiting or active batch uses the selected experts."""

        await self._queue.wait_experts_idle(
            experts,
            monotonic_deadline=monotonic_deadline,
        )

    async def wait_all_idle(self, *, monotonic_deadline: float) -> None:
        """Wait until the Transport waiting area and execution are both empty."""

        await self._queue.wait_all_idle(monotonic_deadline=monotonic_deadline)

    def admitted_count(self, layer_id: int, expert_id: int) -> int:
        """Return waiting plus active batches that name one expert."""

        return self._queue.admitted_count(layer_id, expert_id)

    async def _execute(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            self._record_rejection(TransportErrorCode.UNAVAILABLE.value)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        trace_enabled = self._tracer is not None and self._tracer.current_span_is_recording()
        trace_context = self._tracer.capture_context() if trace_enabled else None
        try:
            with self._trace_span(
                "worker.request.decode",
                attributes={"expertkit.request_bytes": len(payload)},
                enabled=trace_enabled,
            ) as span:
                decoded = await self._run_cpu(decode_request_with_size, payload, self._spec)
                if span is not None:
                    for key, value in _batch_trace_attributes(decoded.batch).items():
                        span.set_attribute(key, value)
        except GrpcProtocolError as error:
            self._record_rejection(TransportErrorCode.PROTOCOL.value)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        del payload

        remaining = context.time_remaining()
        deadline = math.inf if remaining is None else self._clock() + max(0.0, remaining)
        item = _GrpcWorkItem(
            self,
            decoded.batch,
            deadline,
            decoded.retained_tensor_bytes,
            trace_context,
        )
        rejection = await self._admit(item)
        if rejection is not None:
            self._record_rejection(rejection.code.value)
            with self._trace_span("worker.response.encode", enabled=trace_enabled):
                return await self._run_cpu(encode_error_response, rejection, self._spec)

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
            await context.abort(grpc.StatusCode.INTERNAL, "failed to encode Worker response")
            raise AssertionError("context.abort must terminate the handler") from error

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
                expected_slot_count=self._maximum_concurrent_rpcs,
            )
        except GrpcProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        device = self._position_device
        if device is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Worker execution positions are not initialized",
            )
            raise AssertionError("context.abort must terminate the handler")

        async with self._shared_session_lock:
            if description.session_id in self._shared_sessions:
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    "shared-memory session already exists",
                )
                raise AssertionError("context.abort must terminate the handler")
            if len(self._shared_sessions) >= _MAX_SHARED_MEMORY_SESSIONS:
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
                    )
                )
            except (OSError, RuntimeError, ValueError) as error:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"cannot open shared-memory segment: {error}",
                )
                raise AssertionError("context.abort must terminate the handler") from error
            self._shared_sessions[description.session_id] = session
        return encode_open_response()

    async def _execute_shared_memory(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            self._record_rejection(TransportErrorCode.UNAVAILABLE.value)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        trace_enabled = self._tracer is not None and self._tracer.current_span_is_recording()
        trace_context = self._tracer.capture_context() if trace_enabled else None
        try:
            with self._trace_span(
                "worker.request.decode",
                attributes={"expertkit.request_bytes": len(payload)},
                enabled=trace_enabled,
            ) as span:
                request = decode_execute_request(payload, self._spec)
                session = self._shared_sessions.get(request.session_id)
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
        except GrpcProtocolError as error:
            self._record_rejection(TransportErrorCode.PROTOCOL.value)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error

        deadline = (
            math.inf
            if request.timeout_micros == _UINT64_MAX
            else self._clock() + request.timeout_micros / 1_000_000
        )
        item = _GrpcWorkItem(
            self,
            claimed.batch,
            deadline,
            0,
            trace_context,
            output_destination=claimed.output_destination,
            shared_session=session,
            shared_slot_index=claimed.slot_index,
            shared_generation=claimed.generation,
        )
        try:
            rejection = await self._admit(item)
        except BaseException:
            session.release(claimed.slot_index, claimed.generation)
            raise
        if rejection is not None:
            session.release(claimed.slot_index, claimed.generation)
            self._record_rejection(rejection.code.value)
            return encode_execute_error(rejection, self._spec)

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
        except GrpcProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        async with self._shared_session_lock:
            session = self._shared_sessions.get(session_id)
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
            self._shared_sessions.pop(session_id)
            await self._run_cpu(session.close)
        return encode_close_response()

    async def _admit(self, item: _GrpcWorkItem) -> TransportError | None:
        rejection = await self._queue.admit(
            item,
            retained_bytes=item.retained_bytes,
        )
        if rejection is None and self._tracer is not None and item.trace_context is not None:
            item.start_wait_span(self._tracer)
        return rejection

    async def _cancel(self, item: _GrpcWorkItem) -> None:
        item._cancelled = True
        item._cancelled_event.set()
        if await self._queue.cancel_waiting(item):
            item.finish_wait_span("cancelled")
            self._release_shared_slot(item)

    async def _wait_pending_count(self, expected: int) -> None:
        await self._queue.wait_for_pending_count(expected)

    async def _wait_cancelled(self, item: _GrpcWorkItem) -> None:
        await item._cancelled_event.wait()

    async def _complete_success(
        self,
        item: _GrpcWorkItem,
        partial_output: torch.Tensor,
    ) -> None:
        self._require_active(item)
        try:
            if not item.cancelled:
                if partial_output.ndim != 2 or partial_output.shape[0] != item.token_count:
                    raise ValueError("partial output token count does not match the received batch")
                with self._trace_span(
                    "worker.response.encode",
                    enabled=item.trace_context is not None,
                ):
                    if item.output_destination is None:
                        payload = await self._run_cpu(
                            encode_success_response,
                            partial_output,
                            self._spec,
                        )
                    else:
                        if partial_output.data_ptr() != item.output_destination.data_ptr():
                            raise ValueError(
                                "shared-memory response did not use its registered destination"
                            )
                        if item.shared_generation is None:
                            raise RuntimeError("shared-memory response has no generation")
                        payload = encode_execute_success(item.shared_generation)
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
        item: _GrpcWorkItem,
        error: TransportError,
    ) -> None:
        self._require_active(item)
        try:
            if not item.cancelled:
                status = _NATIVE_ERROR_STATUS.get(error.code)
                if status is None:
                    with self._trace_span(
                        "worker.response.encode",
                        enabled=item.trace_context is not None,
                    ):
                        if item.shared_session is None:
                            payload = await self._run_cpu(
                                encode_error_response,
                                error,
                                self._spec,
                            )
                        else:
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

    async def _finish_active(self, item: _GrpcWorkItem) -> None:
        item._batch = None
        cleanup = asyncio.create_task(self._release_active(item))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _release_active(self, item: _GrpcWorkItem) -> None:
        self._release_shared_slot(item)
        await self._queue.finish(item)

    @staticmethod
    def _release_shared_slot(item: _GrpcWorkItem) -> None:
        if item.shared_session is None:
            return
        if item.shared_slot_index is None or item.shared_generation is None:
            raise RuntimeError("shared-memory item has incomplete slot ownership")
        item.shared_session.release(item.shared_slot_index, item.shared_generation)

    def _require_active(self, item: _GrpcWorkItem) -> None:
        self._queue.require_active(item)

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

    def _record_rejection(self, reason: str) -> None:
        with suppress(Exception):
            self._on_rejection(reason)

    async def _close(self) -> None:
        self._closing = True
        if self._server is not None:
            await self._server.stop(None)
        waiting = await self._queue.begin_close()
        for item in waiting:
            item._cancelled = True
            item._cancelled_event.set()
            item.finish_wait_span("closed")
            self._release_shared_slot(item)
        await self._queue.wait_active_empty()
        async with self._shared_session_lock:
            sessions = tuple(self._shared_sessions.values())
            self._shared_sessions.clear()
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
