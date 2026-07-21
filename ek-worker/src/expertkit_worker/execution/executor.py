"""Bounded Worker execution that takes batches directly from Transport."""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial

import structlog
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.tracing import Tracer, TraceSpan
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    ReceiverClosed,
    WorkerBatchReceiver,
)

from expertkit_worker.backends import (
    BackendFatalError,
    BackendRequestError,
    BackendWeightUnavailable,
    ComputeBackend,
    InvalidBackendInput,
    UnsupportedBackendBatch,
)
from expertkit_worker.execution.slot import ExecutionResult, ExecutionSlot
from expertkit_worker.observability.api import NoopWorkerMetrics, WorkerMetrics

logger = structlog.get_logger(__name__)


def _request_error(error: BackendRequestError) -> TransportError:
    if isinstance(error, BackendWeightUnavailable):
        return TransportError(
            TransportErrorCode.EXPERT_NOT_READY,
            retryable=True,
            unavailable_expert_ids=error.unavailable_expert_ids,
            diagnostic=str(error),
        )
    if isinstance(error, UnsupportedBackendBatch):
        return TransportError(
            TransportErrorCode.UNSUPPORTED,
            retryable=False,
            diagnostic=str(error),
        )
    if isinstance(error, InvalidBackendInput):
        return TransportError(
            TransportErrorCode.INVALID_REQUEST,
            retryable=False,
            diagnostic=str(error),
        )
    return TransportError(
        TransportErrorCode.INVALID_REQUEST,
        retryable=False,
        diagnostic="Backend rejected the request without a more specific classification",
    )


class WorkerExecutor:
    """Drive a fixed number of active slots without another waiting queue."""

    def __init__(
        self,
        receiver: WorkerBatchReceiver,
        backend: ComputeBackend,
        *,
        instance_id: int,
        buffer_config: BatchBufferConfig,
        slot_count: int,
        clock: Callable[[], float] = time.monotonic,
        metrics: WorkerMetrics | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")
        if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count <= 0:
            raise ValueError("slot_count must be a positive integer")
        backend.capabilities.validate_runtime(
            max_batch_tokens=buffer_config.max_batch_tokens,
            active_batches=slot_count,
        )

        self._receiver = receiver
        self._backend = backend
        self._instance_id = instance_id
        self._clock = clock
        self._metrics = metrics or NoopWorkerMetrics()
        self._tracer = tracer
        self._slots: list[ExecutionSlot] = []
        try:
            for _ in range(slot_count):
                buffers = receiver.create_batch_buffers(buffer_config)
                self._slots.append(
                    ExecutionSlot(
                        buffer_config,
                        buffers,
                        enable_cuda_timing=tracer is not None,
                    )
                )
        except BaseException:
            for slot in self._slots:
                slot.close()
            raise
        self._executor = ThreadPoolExecutor(
            max_workers=slot_count,
            thread_name_prefix="expertkit-worker-execution",
        )
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._fatal_error: BackendFatalError | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def fixed_device_bytes(self) -> int:
        """Return device bytes reserved by all active input and output slots."""

        return sum(slot.device_bytes for slot in self._slots)

    @property
    def fixed_host_staging_bytes(self) -> int:
        """Return Host staging bytes reserved by all active slots."""

        return sum(slot.host_staging_bytes for slot in self._slots)

    async def start(self) -> None:
        """Start the receiver and one loop for each execution slot."""

        if self._closed:
            raise RuntimeError("Worker execution is closed")
        if self._tasks:
            return
        await self._receiver.start()
        self._tasks = tuple(
            asyncio.create_task(
                self._serve_slot(slot),
                name=f"worker-active-slot-{index}",
            )
            for index, slot in enumerate(self._slots)
        )

    async def wait(self) -> None:
        """Wait for all slot loops and re-raise the first fatal Backend error."""

        if not self._tasks:
            raise RuntimeError("Worker execution has not been started")
        await asyncio.gather(*self._tasks)
        if self._fatal_error is not None:
            raise self._fatal_error

    async def close(self) -> None:
        """Stop Transport, finish active work, and release fixed resources."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._receiver.close()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._executor.shutdown, wait=True, cancel_futures=True),
        )
        for slot in self._slots:
            slot.close()

    async def _serve_slot(self, slot: ExecutionSlot) -> None:
        while True:
            try:
                received = await self._receiver.receive()
            except ReceiverClosed:
                return
            started_at = self._clock()
            self._metrics.batch_started()
            fatal: BackendFatalError | None = None
            failed = True
            try:
                fatal = await self._process(received, slot)
                failed = fatal is not None
            finally:
                self._metrics.batch_finished(
                    fatal=failed,
                    duration_seconds=self._clock() - started_at,
                )
            if fatal is not None:
                if self._fatal_error is None:
                    self._fatal_error = fatal
                    logger.critical(
                        "worker_backend_fatal",
                        error_code=fatal.reason.value,
                        diagnostic=fatal.diagnostic,
                    )
                await self._receiver.close()
                return

    async def _process(
        self,
        received: ReceivedBatch,
        slot: ExecutionSlot,
    ) -> BackendFatalError | None:
        if self._tracer is None or received.trace_context is None:
            return await self._process_batch(received, slot, None)
        batch = received.batch
        attributes = {
            "expertkit.instance_id": batch.instance_id,
            "expertkit.layer_id": batch.layer_id,
            "expertkit.topology_version": batch.topology_version,
            "expertkit.token_count": batch.token_count,
            "expertkit.assignment_count": batch.token_count * batch.top_k,
            "expertkit.backend": type(self._backend).__name__,
            "expertkit.device": str(slot.device),
        }
        span_context = self._tracer.start_as_current_span(
            "worker.batch.execute",
            context=received.trace_context,
            attributes=attributes,
        )
        with span_context as span:
            return await self._process_batch(received, slot, span)

    async def _process_batch(
        self,
        received: ReceivedBatch,
        slot: ExecutionSlot,
        span: TraceSpan | None,
    ) -> BackendFatalError | None:
        try:
            batch = received.batch
            if batch.instance_id != self._instance_id:
                received.release_input()
                await received.reject(
                    TransportError(
                        TransportErrorCode.INVALID_REQUEST,
                        retryable=False,
                        diagnostic="Worker batch instance ID does not match this process",
                    )
                )
                self._metrics.batch_rejected(TransportErrorCode.INVALID_REQUEST.value)
                return None
            result = await self._run_slot(slot, received, span)
        except BackendRequestError as error:
            rejection = _request_error(error)
            logger.warning(
                "worker_computation_rejected",
                error_code=rejection.code.value,
                unavailable_expert_ids=rejection.unavailable_expert_ids,
            )
            self._metrics.batch_rejected(rejection.code.value)
            await received.reject(rejection)
            return None
        except BackendFatalError as error:
            with suppress(Exception):
                await received.reject(
                    TransportError(
                        TransportErrorCode.UNAVAILABLE,
                        retryable=True,
                        diagnostic="Worker Backend cannot continue serving",
                    )
                )
            return error

        try:
            if result.rejection is not None:
                self._metrics.batch_rejected(result.rejection.code.value)
                await received.reject(result.rejection)
            elif result.output is None:
                raise RuntimeError("active slot produced neither output nor rejection")
            else:
                await received.complete(result.output)
        except Exception:
            self._metrics.batch_rejected("response_failed")
            logger.error("worker_computation_response_failed", exc_info=True)
        finally:
            result.release()
        return None

    async def _run_slot(
        self,
        slot: ExecutionSlot,
        received: ReceivedBatch,
        span: TraceSpan | None,
    ) -> ExecutionResult:
        loop = asyncio.get_running_loop()
        tracer = self._tracer if span is not None else None
        execute = partial(
            slot.execute,
            received,
            self._backend,
            clock=self._clock,
            tracer=tracer,
            batch_span=span,
        )
        if tracer is None:
            work = loop.run_in_executor(self._executor, execute)
        else:
            context = contextvars.copy_context()
            work = loop.run_in_executor(self._executor, context.run, execute)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            try:
                result = await work
            except BaseException:
                pass
            else:
                result.release()
            raise
