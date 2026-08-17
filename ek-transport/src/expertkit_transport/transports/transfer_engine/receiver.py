"""Worker receiver for pull-based Mooncake Transfer Engine requests."""

from __future__ import annotations

import asyncio
import math
import secrets
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
from expertkit_transport.transports.queue import ReceiverQueue
from expertkit_transport.transports.transfer_engine.arena import (
    TransferArena,
    TransferArenaSlot,
    transfer_lengths,
)
from expertkit_transport.transports.transfer_engine.codec import (
    TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES,
    TransferCloseSession,
    TransferExecutePull,
    TransferOpenSessionResult,
    decode_close_session,
    decode_execute_pull_request,
    decode_open_session_request,
    encode_admitted,
    encode_error,
    encode_open_session_response,
    encode_success,
)
from expertkit_transport.transports.transfer_engine.runtime import (
    TransferEngineRuntimeProtocol,
)
from expertkit_transport.transports.transfer_engine.session import TransferEngineSession
from expertkit_transport.transports.transfer_engine.worker_buffers import (
    TransferEngineWorkerBatchBuffers,
)

_OPEN_METHOD_NAME = "OpenSession"
_EXECUTE_METHOD_NAME = "ExecutePull"
_CLOSE_METHOD_NAME = "CloseSession"
_SERVICE_NAME = "ek.worker.v2.TransferEngineComputationService"
_UINT64_MAX = (1 << 64) - 1
_MAX_SESSION_EPOCHS = 4096


def _identity(payload: bytes) -> bytes:
    return payload


async def _await_task_terminal(task: asyncio.Task[Any]) -> Any:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            continue


def _batch_trace_attributes(batch: WorkerBatch) -> dict[str, str | int]:
    return {
        "expertkit.instance_id": batch.instance_id,
        "expertkit.layer_id": batch.layer_id,
        "expertkit.topology_version": batch.topology_version,
        "expertkit.token_count": batch.token_count,
        "expertkit.assignment_count": batch.token_count * batch.top_k,
        "expertkit.transport": "transfer_engine",
    }


class _InputTransferFailed(RuntimeError):
    """Tell ``receive`` to skip an item whose native pull failed."""


class _TransferReceivedBatch(ReceivedBatch):
    def __init__(
        self,
        owner: TransferEngineWorkerBatchReceiver,
        request: TransferExecutePull,
        session: TransferEngineSession,
        slot: TransferArenaSlot,
        batch: WorkerBatch,
        monotonic_deadline: float,
        trace_context: TraceContext | None,
    ) -> None:
        self._owner = owner
        self.request = request
        self.session = session
        self.slot = slot
        self._batch: WorkerBatch | None = batch
        self._deadline = monotonic_deadline
        self._trace_context = trace_context
        self._wait_span: TraceSpan | None = None
        self._cancelled = False
        self._input_ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._response: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._released_event = asyncio.Event()
        self.released = False

    @property
    def trace_context(self) -> TraceContext | None:
        return self._trace_context

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("received Transfer Engine input has already been released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def output_destination(self) -> torch.Tensor:
        return self.slot.partial_output[: self.request.token_count]

    def release_input(self) -> None:
        self._owner._require_active(self)
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        await self._owner._complete_success(self, partial_output)

    async def reject(self, error: TransportError) -> None:
        await self._owner._complete_error(self, error)

    async def reject_unsafe(self, error: TransportError) -> None:
        await self._owner._complete_unsafe_error(self, error)

    def start_wait_span(self, tracer: Tracer) -> None:
        if self._wait_span is not None:
            raise RuntimeError("Transfer Engine waiting span is already active")
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


class TransferEngineWorkerBatchReceiver(WorkerBatchReceiver):
    """Admit metadata over gRPC and pull registered Tensor regions with Mooncake."""

    def __init__(
        self,
        control_listen: str,
        endpoint_config: WorkerEndpointConfig,
        *,
        runtime: TransferEngineRuntimeProtocol,
        worker_start_id: str,
        max_active_batches: int,
        max_pending_batches: int,
        session_close_grace_secs: float = 30.0,
        owns_runtime: bool = True,
        clock: Callable[[], float] = time.monotonic,
        interceptors: Sequence[grpc.aio.ServerInterceptor] = (),
        tracer: Tracer | None = None,
        on_rejection: Callable[[str], None] | None = None,
        on_pending_changed: Callable[[int], None] | None = None,
    ) -> None:
        if not control_listen:
            raise ValueError("control_listen must not be empty")
        if not isinstance(worker_start_id, str) or not worker_start_id:
            raise ValueError("worker_start_id must not be empty")
        for name, value in (
            ("max_active_batches", max_active_batches),
            ("max_pending_batches", max_pending_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(owns_runtime, bool):
            raise ValueError("owns_runtime must be a Boolean")
        if (
            isinstance(session_close_grace_secs, bool)
            or not isinstance(session_close_grace_secs, int | float)
            or session_close_grace_secs < 0
        ):
            raise ValueError("session_close_grace_secs must be nonnegative")
        self._listen = control_listen
        self._spec = endpoint_config
        self._runtime = runtime
        self._worker_start_id = worker_start_id
        self._owns_runtime = owns_runtime
        self._session_close_grace_secs = float(session_close_grace_secs)
        self._slot_count = max_active_batches + max_pending_batches
        self._arena = TransferArena(
            endpoint_config,
            device=runtime.device,
            capacity=self._slot_count,
        )
        self._registered = False
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
        self._sessions: dict[str, TransferEngineSession] = {}
        self._session_nonces: dict[str, tuple[str, str, str]] = {}
        # Active target ownership plus RDMA endpoint-generation tombstones.
        # Descriptor-only retirement deliberately preserves shared QPs, so an
        # endpoint must not be rebound to a different remote process generation
        # until this Worker runtime itself restarts.
        self._target_generations: dict[str, str] = {}
        self._target_close_gates: dict[str, set[str]] = {}
        self._target_commit_locks: dict[str, asyncio.Lock] = {}
        self._session_lock = asyncio.Lock()
        self._sessions_empty = asyncio.Event()
        self._sessions_empty.set()
        self._server: grpc.aio.Server | None = None
        self._start_lock = asyncio.Lock()
        self._execution_device: torch.device | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None
        self._drive_tasks: set[asyncio.Task[bytes]] = set()
        self._quarantined_slots: set[TransferArenaSlot] = set()

    @property
    def bound_port(self) -> int:
        if self._bound_port is None:
            raise RuntimeError("Transfer Engine receiver has not been started")
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

    @property
    def worker_start_id(self) -> str:
        return self._worker_start_id

    @property
    def fixed_device_bytes(self) -> int:
        """Return the registered receive arena charged to the Worker budget."""

        return self._arena.slab.numel() * self._arena.slab.element_size()

    async def start(self) -> None:
        async with self._start_lock:
            if self._closing:
                raise RuntimeError("Transfer Engine receiver is closing")
            if self._server is not None:
                return
            await self._runtime.start()
            await self._runtime.register_tensor(self._arena.slab)
            self._registered = True
            options = (
                ("grpc.max_receive_message_length", TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES),
                ("grpc.max_send_message_length", TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES),
            )
            server = grpc.aio.server(
                options=options,
                maximum_concurrent_rpcs=self._slot_count + 16,
                interceptors=self._interceptors,
            )
            service = grpc.method_handlers_generic_handler(
                _SERVICE_NAME,
                {
                    _OPEN_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
                        self._open_session,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    _EXECUTE_METHOD_NAME: grpc.unary_stream_rpc_method_handler(
                        self._execute_pull,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                    _CLOSE_METHOD_NAME: grpc.unary_unary_rpc_method_handler(
                        self._close_session,
                        request_deserializer=_identity,
                        response_serializer=_identity,
                    ),
                },
            )
            server.add_generic_rpc_handlers((service,))
            port = server.add_insecure_port(self._listen)
            if port == 0:
                await self._runtime.unregister_tensor(self._arena.slab)
                self._registered = False
                raise RuntimeError(
                    "Transfer Engine receiver could not bind its control listen address"
                )
            try:
                await server.start()
            except BaseException:
                await self._runtime.unregister_tensor(self._arena.slab)
                self._registered = False
                raise
            self._server = server
            self._bound_port = port

    async def receive(self) -> ReceivedBatch:
        while True:
            item = await self._queue.take()
            assert isinstance(item, _TransferReceivedBatch)
            item.finish_wait_span("active")
            try:
                await asyncio.shield(item._input_ready)
            except _InputTransferFailed:
                cleanup = asyncio.create_task(self._finish_failed_active(item))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    with suppress(BaseException):
                        await _await_task_terminal(cleanup)
                    raise
                continue
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(self._cancel_active_take(item))
                with suppress(BaseException):
                    await _await_task_terminal(cleanup)
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
            raise ValueError(
                "Worker buffer shape/device does not match the Transfer Engine endpoint"
            )
        if self._execution_device is None:
            self._execution_device = spec.device
        elif self._execution_device != spec.device:
            raise ValueError("all Worker execution slots must use the same device")
        return TransferEngineWorkerBatchBuffers(
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
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    def admitted_count(self, layer_id: int, expert_id: int) -> int:
        return self._queue.admitted_count(layer_id, expert_id)

    async def _open_session(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        try:
            self._runtime.ensure_healthy()
            request = decode_open_session_request(
                payload,
                self._spec,
                self._worker_start_id,
                self._runtime.backend,
            )
            async with self._session_lock:
                if self._closing:
                    await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
                existing = self._sessions.get(request.client_epoch)
                if request.client_session_id in self._target_close_gates:
                    raise TransportProtocolError(
                        "Transfer Engine remote target is gated for two-phase close"
                    )
                if existing is None:
                    if request.client_epoch in self._session_nonces:
                        raise TransportProtocolError(
                            "Transfer Engine client epoch is closed and cannot be replayed"
                        )
                    if len(self._session_nonces) >= _MAX_SESSION_EPOCHS:
                        raise TransportProtocolError(
                            "Transfer Engine session epoch capacity is exhausted"
                        )
                    target_generation = self._target_generations.get(request.client_session_id)
                    if (
                        target_generation is not None
                        and target_generation != request.client_runtime_generation
                    ):
                        raise TransportProtocolError(
                            "Transfer Engine remote endpoint belongs to a stale runtime "
                            "generation; restart the Worker"
                        )
                    session = TransferEngineSession(
                        client_epoch=request.client_epoch,
                        session_nonce=secrets.token_hex(32),
                        target_session_id=request.client_session_id,
                        target_runtime_generation=request.client_runtime_generation,
                        backend=request.backend,
                        arena=request.arena,
                    )
                    self._sessions[request.client_epoch] = session
                    self._sessions_empty.clear()
                    self._session_nonces[request.client_epoch] = (
                        session.session_nonce,
                        session.target_session_id,
                        session.target_runtime_generation,
                    )
                    self._target_generations[request.client_session_id] = (
                        request.client_runtime_generation
                    )
                else:
                    session = existing
                    if (
                        existing.target_session_id != request.client_session_id
                        or existing.target_runtime_generation != request.client_runtime_generation
                        or existing.backend != request.backend
                        or existing.arena != request.arena
                    ):
                        raise TransportProtocolError(
                            "Transfer Engine client epoch changed its registered session"
                        )
        except TransportProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        except TransportError as error:
            await context.abort(grpc.StatusCode.UNAVAILABLE, error.diagnostic)
            raise AssertionError("context.abort must terminate the handler") from error
        return encode_open_session_response(
            TransferOpenSessionResult(
                worker_start_id=self._worker_start_id,
                worker_session_id=self._runtime.session_id,
                session_nonce=session.session_nonce,
                backend=self._runtime.backend,
                arena=self._arena.descriptor,
            ),
            self._spec,
        )

    async def _close_session(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        try:
            request = decode_close_session(payload, self._worker_start_id)
            if request.phase == "prepare":
                async with self._session_lock:
                    session = self._require_close_session_locked(request)
                    await self._prepare_session_close_locked(session)
            else:
                await self._commit_session_close(request)
        except TransportProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        return b""

    async def _execute_pull(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> Any:
        if self._closing:
            self._record_rejection(TransportErrorCode.UNAVAILABLE.value)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        trace_enabled = self._tracer is not None and self._tracer.current_span_is_recording()
        trace_context = self._tracer.capture_context() if trace_enabled else None
        target_gated = False
        try:
            request = decode_execute_pull_request(
                payload,
                self._spec,
                self._worker_start_id,
            )
            try:
                self._runtime.ensure_healthy()
            except TransportError as error:
                self._record_rejection(error.code.value)
                yield encode_error(request.sequence, error)
                return
            async with self._session_lock:
                if self._closing:
                    self._record_rejection(TransportErrorCode.UNAVAILABLE.value)
                    await context.abort(
                        grpc.StatusCode.UNAVAILABLE,
                        "Worker is shutting down",
                    )
                session = self._sessions.get(request.client_epoch)
                if session is None:
                    raise TransportProtocolError("Transfer Engine session is unknown or closed")
                if not secrets.compare_digest(session.session_nonce, request.session_nonce):
                    raise TransportProtocolError("Transfer Engine session nonce is stale")
                target_gated = session.target_session_id in self._target_close_gates
                if not target_gated:
                    session.claim(
                        sequence=request.sequence,
                        slot_index=request.client_slot_index,
                        generation=request.client_slot_generation,
                    )
        except TransportProtocolError as error:
            self._record_rejection(TransportErrorCode.PROTOCOL.value)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error
        del payload

        if target_gated:
            error = TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="Transfer Engine remote target is gated for two-phase close",
            )
            self._record_rejection(error.code.value)
            yield encode_error(request.sequence, error)
            return

        slot = self._arena.take()
        if slot is None:
            session.release(request.sequence)
            error = TransportError(
                TransportErrorCode.BUSY,
                retryable=True,
                diagnostic="Worker Transfer Engine receive slots are full",
            )
            self._record_rejection(error.code.value)
            yield encode_error(request.sequence, error)
            return
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
        token_count = request.token_count
        batch = WorkerBatch(
            instance_id=self._spec.instance_id,
            layer_id=request.layer_id,
            topology_version=request.topology_version,
            hidden_states=slot.hidden_states[:token_count],
            token_indices=None,
            expert_ids=slot.expert_ids[:token_count],
            routing_weights=slot.routing_weights[:token_count],
            distinct_expert_ids=request.distinct_expert_ids,
        )
        item = _TransferReceivedBatch(
            self,
            request,
            session,
            slot,
            batch,
            deadline,
            trace_context,
        )
        try:
            rejection = await self._queue.admit(item, retained_bytes=0)
        except BaseException:
            self._release_item(item)
            raise
        if rejection is not None:
            self._release_item(item)
            self._record_rejection(rejection.code.value)
            yield encode_error(request.sequence, rejection)
            return
        if self._tracer is not None and item.trace_context is not None:
            item.start_wait_span(self._tracer)

        drive = asyncio.create_task(
            self._drive_request(item),
            name=f"transfer-engine-worker-sequence-{request.sequence}",
        )
        self._drive_tasks.add(drive)
        drive.add_done_callback(self._drive_tasks.discard)
        try:
            yield encode_admitted(request.sequence)
            terminal = await asyncio.shield(drive)
            yield terminal
        except (asyncio.CancelledError, GeneratorExit):
            cleanup = asyncio.create_task(self._cancel(item))
            with suppress(BaseException):
                await _await_task_terminal(cleanup)
            with suppress(BaseException):
                await _await_task_terminal(drive)
            return

    async def _drive_request(self, item: _TransferReceivedBatch) -> bytes:
        request = item.request
        token_count = request.token_count
        remote = item.session.arena.addresses(request.client_slot_index)
        lengths = transfer_lengths(self._spec, token_count)
        try:
            await self._runtime.batch_read(
                item.session.target_session_id,
                (
                    item.slot.hidden_states[:token_count],
                    item.slot.expert_ids[:token_count],
                    item.slot.routing_weights[:token_count],
                ),
                remote[:3],
                lengths[:3],
                monotonic_deadline=item.monotonic_deadline,
            )
            await self._runtime.acquire_remote_writes(
                monotonic_deadline=item.monotonic_deadline,
            )
            if not item._input_ready.done():
                item._input_ready.set_result(None)
            return await asyncio.shield(item._response)
        except BaseException as error:
            try:
                self._runtime.ensure_healthy()
            except TransportError:
                self._quarantined_slots.add(item.slot)
            if not item._input_ready.done():
                await self._fail_input_transfer(item, error)
            if not item.released:
                await item._released_event.wait()
            if isinstance(error, TransportError):
                return encode_error(request.sequence, error)
            raise

    async def _cancel(self, item: _TransferReceivedBatch) -> None:
        item._cancelled = True
        if await self._queue.cancel_waiting(item):
            item.finish_wait_span("cancelled")
            await self._finish_cancelled_waiting(item)

    async def _finish_cancelled_waiting(self, item: _TransferReceivedBatch) -> None:
        terminal = encode_error(
            item.request.sequence,
            TransportError(
                TransportErrorCode.CANCELLED,
                retryable=False,
                diagnostic="Transfer Engine caller cancelled before Worker execution",
            ),
        )
        try:
            with suppress(_InputTransferFailed):
                await asyncio.shield(item._input_ready)
        finally:
            self._release_item(item)
        if not item._response.done():
            item._response.set_result(terminal)

    async def _cancel_active_take(self, item: _TransferReceivedBatch) -> None:
        item._cancelled = True
        try:
            await asyncio.shield(item._input_ready)
        except _InputTransferFailed:
            await self._finish_failed_active(item)
            return
        terminal = encode_error(
            item.request.sequence,
            TransportError(
                TransportErrorCode.CANCELLED,
                retryable=False,
                diagnostic="Worker receive loop stopped before execution",
            ),
        )
        item._batch = None
        await self._queue.finish(item)
        self._release_item(item)
        if not item._response.done():
            item._response.set_result(terminal)

    async def _complete_success(
        self,
        item: _TransferReceivedBatch,
        partial_output: torch.Tensor,
    ) -> None:
        self._require_active(item)
        try:
            destination = item.output_destination
            if partial_output.shape != destination.shape:
                raise ValueError("partial output shape does not match the Transfer Engine request")
            if (
                partial_output.dtype != self._spec.dtype
                or partial_output.device != self._runtime.device
                or partial_output.data_ptr() != destination.data_ptr()
            ):
                raise ValueError("Transfer Engine response did not use its registered output slot")
            remote_output = item.session.arena.addresses(item.request.client_slot_index)[3]
            output_bytes = transfer_lengths(self._spec, item.request.token_count)[3]
            await self._runtime.batch_write(
                item.session.target_session_id,
                (partial_output,),
                (remote_output,),
                (output_bytes,),
                # Once compute has produced an answer, keep both registered slots
                # leased until native DMA is terminal, even after caller timeout.
                monotonic_deadline=math.inf,
            )
        except BaseException as error:
            try:
                self._runtime.ensure_healthy()
            except TransportError:
                # An unexpected native WRITE exception may leave DMA reading
                # this registered receive slot.  A quarantined runtime is the
                # signal that native terminal state was not proven.
                self._quarantined_slots.add(item.slot)
            await self._finish_active(item)
            if not item._response.done():
                if isinstance(error, TransportError):
                    item._response.set_result(encode_error(item.request.sequence, error))
                else:
                    item._response.set_exception(error)
            raise
        else:
            await self._finish_active(item)
            if not item._response.done():
                item._response.set_result(encode_success(item.request.sequence))

    async def _complete_error(
        self,
        item: _TransferReceivedBatch,
        error: TransportError,
    ) -> None:
        self._require_active(item)
        await self._finish_active(item)
        if not item._response.done():
            item._response.set_result(encode_error(item.request.sequence, error))

    async def _complete_unsafe_error(
        self,
        item: _TransferReceivedBatch,
        error: TransportError,
    ) -> None:
        self._require_active(item)
        diagnostic = (
            error.diagnostic
            or "Worker device work may still reference the registered receive arena"
        )
        self._runtime.quarantine(
            f"Unsafe Worker execution retained the Transfer Engine receive arena: {diagnostic}"
        )
        self._quarantined_slots.add(item.slot)
        terminal = TransportError(
            error.code,
            retryable=False,
            observed_topology_version=error.observed_topology_version,
            min_topology_version=error.min_topology_version,
            unavailable_expert_ids=error.unavailable_expert_ids,
            diagnostic=diagnostic,
        )
        await self._finish_active(item)
        if not item._response.done():
            item._response.set_result(encode_error(item.request.sequence, terminal))

    async def _finish_active(self, item: _TransferReceivedBatch) -> None:
        item._batch = None
        cleanup = asyncio.create_task(self._release_active(item))
        # Releasing the queue/session/slot is a commit point. Consume repeated
        # cancellation until it completes so the caller can publish the ordered
        # terminal response instead of leaving _drive_request waiting forever.
        await _await_task_terminal(cleanup)

    async def _release_active(self, item: _TransferReceivedBatch) -> None:
        await self._queue.finish(item)
        self._release_item(item)

    async def _finish_failed_active(self, item: _TransferReceivedBatch) -> None:
        await self._queue.finish(item)
        self._release_item(item)

    async def _fail_input_transfer(
        self,
        item: _TransferReceivedBatch,
        error: BaseException,
    ) -> None:
        marker = _InputTransferFailed(str(error))
        marker.__cause__ = error
        if not item._input_ready.done():
            item._input_ready.set_exception(marker)
        if await self._queue.cancel_waiting(item):
            item.finish_wait_span("receive_failed")
            with suppress(BaseException):
                item._input_ready.exception()
            self._release_item(item)

    def _release_item(self, item: _TransferReceivedBatch) -> None:
        if item.released:
            return
        item.released = True
        item._batch = None
        if item.slot not in self._quarantined_slots:
            self._arena.put(item.slot)
        item.session.release(item.request.sequence)
        item._released_event.set()

    async def _prepare_session_close_locked(self, session: TransferEngineSession) -> None:
        current = self._sessions.get(session.client_epoch)
        if current is not session:
            raise RuntimeError("Transfer Engine close prepare session is not active")

        gates = self._target_close_gates.setdefault(session.target_session_id, set())
        gates.add(session.client_epoch)
        session.begin_close_prepare()

        target_sessions = tuple(
            candidate
            for candidate in self._sessions.values()
            if candidate.target_session_id == session.target_session_id
        )
        await asyncio.gather(*(candidate.wait_idle() for candidate in target_sessions))
        session.finish_close_prepare()

    def _require_close_session_locked(
        self,
        request: TransferCloseSession,
    ) -> TransferEngineSession:
        tombstone = self._session_nonces.get(request.client_epoch)
        if (
            tombstone is None
            or not secrets.compare_digest(tombstone[0], request.session_nonce)
            or tombstone[1] != request.client_session_id
            or tombstone[2] != request.client_runtime_generation
        ):
            raise TransportProtocolError(
                "Transfer Engine CloseSession nonce or target generation is stale"
            )
        session = self._sessions.get(request.client_epoch)
        if session is None:
            raise TransportProtocolError("Transfer Engine close session is already retired")
        return session

    async def _commit_session_close(self, request: TransferCloseSession) -> None:
        async with self._session_lock:
            tombstone = self._session_nonces.get(request.client_epoch)
            if (
                tombstone is None
                or not secrets.compare_digest(tombstone[0], request.session_nonce)
                or tombstone[1] != request.client_session_id
                or tombstone[2] != request.client_runtime_generation
            ):
                raise TransportProtocolError(
                    "Transfer Engine CloseSession nonce or target generation is stale"
                )
            if request.client_epoch not in self._sessions:
                return
            commit_lock = self._target_commit_locks.setdefault(
                request.client_session_id,
                asyncio.Lock(),
            )

        async with commit_lock:
            async with self._session_lock:
                if request.client_epoch not in self._sessions:
                    return
                session = self._require_close_session_locked(request)
                gates = self._target_close_gates.get(session.target_session_id)
                if not session.close_prepared or gates is None or session.client_epoch not in gates:
                    raise TransportProtocolError(
                        "Transfer Engine close commit has no matching prepared target gate"
                    )
                target_sessions = tuple(
                    candidate
                    for candidate in self._sessions.values()
                    if candidate.target_session_id == session.target_session_id
                )
                if any(not candidate.idle for candidate in target_sessions):
                    raise RuntimeError(
                        "Transfer Engine prepared target acquired work before close commit"
                    )
                remaining_target_sessions = tuple(
                    candidate for candidate in target_sessions if candidate is not session
                )
                if not remaining_target_sessions:
                    generation = self._target_generations.get(session.target_session_id)
                    if generation != session.target_runtime_generation:
                        raise RuntimeError(
                            "Transfer Engine target generation ownership is corrupted"
                        )

            # Keep the target gate visible while backend-specific retirement
            # commits so sibling Execute/Open calls cannot reopen stale state.
            # Intra-NVLink and RDMA invalidate backend-specific remote-target
            # caches here, after the owner synchronously deregistered its arena.
            try:
                await self._runtime.invalidate_remote_session(session.target_session_id)
            except BaseException:
                self._runtime.quarantine(
                    "Transfer Engine could not commit drained remote-target retirement"
                )
                raise

            async with self._session_lock:
                current = self._require_close_session_locked(request)
                if current is not session:
                    raise RuntimeError("Transfer Engine close session ownership changed")
                gates = self._target_close_gates[session.target_session_id]
                self._sessions.pop(session.client_epoch)
                if not self._sessions:
                    self._sessions_empty.set()
                if not remaining_target_sessions and session.backend != "rdma":
                    self._target_generations.pop(session.target_session_id)
                gates.remove(session.client_epoch)
                if not gates:
                    self._target_close_gates.pop(session.target_session_id)
                    self._target_commit_locks.pop(session.target_session_id, None)

    def _require_active(self, item: _TransferReceivedBatch) -> None:
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
        waiting = await self._queue.begin_close()
        for received in waiting:
            assert isinstance(received, _TransferReceivedBatch)
            received._cancelled = True
            received.finish_wait_span("closed")
            await self._finish_cancelled_waiting(received)
        await self._queue.wait_active_empty()
        if self._drive_tasks:
            await asyncio.gather(*tuple(self._drive_tasks), return_exceptions=True)

        # Route removal makes Frontends close their per-Worker transports. Keep
        # only CloseSession reachable during that propagation window: Open and
        # ExecutePull reject on _closing, while CloseSession drains sibling work
        # and invalidates Mooncake's target-wide CUDA IPC cache before ACKing.
        session_barrier_failed = False
        async with self._session_lock:
            sessions_pending = bool(self._sessions)
        if sessions_pending:
            try:
                async with asyncio.timeout(self._session_close_grace_secs):
                    await self._sessions_empty.wait()
            except TimeoutError:
                session_barrier_failed = True
                self._runtime.quarantine(
                    "Transfer Engine Worker shutdown expired before every Frontend "
                    "proved CloseSession cache invalidation"
                )

        if self._server is not None:
            if session_barrier_failed:
                await self._server.stop(None)
            else:
                # Let the final CloseSession handler deliver its empty ACK before
                # shutting down the listener that carried the memory barrier.
                await self._server.stop(self._session_close_grace_secs)

        retain_arena = session_barrier_failed or bool(self._quarantined_slots)
        if not retain_arena:
            self._arena.close()
        if self._registered and not retain_arena:
            await self._runtime.unregister_tensor(self._arena.slab)
            self._registered = False
        if not session_barrier_failed:
            self._sessions.clear()
            self._session_nonces.clear()
            self._target_generations.clear()
            self._target_close_gates.clear()
            self._target_commit_locks.clear()
        if self._owns_runtime:
            await self._runtime.close()


__all__ = ["TransferEngineWorkerBatchReceiver"]
