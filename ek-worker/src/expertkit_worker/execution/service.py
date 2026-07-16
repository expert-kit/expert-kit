"""Bounded execution service that takes batches directly from Transport."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial

import structlog
from expertkit_transport.contracts import (
    ReceivedWorkerBatch,
    ReceiverClosed,
    TransportError,
    TransportErrorCode,
    WorkerBatchReceiver,
    WorkerPositionSpec,
)

from expertkit_worker.backends import (
    BackendFatalError,
    BackendRequestError,
    BackendWeightUnavailable,
    ComputeBackend,
    InvalidBackendInput,
    UnsupportedBackendBatch,
)
from expertkit_worker.execution.position import ActivePosition, PositionResult

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


class WorkerExecution:
    """Drive a fixed number of active positions without another waiting queue."""

    def __init__(
        self,
        receiver: WorkerBatchReceiver,
        backend: ComputeBackend,
        *,
        instance_id: int,
        position_spec: WorkerPositionSpec,
        active_positions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
            raise ValueError("instance_id must be a positive integer")
        if (
            isinstance(active_positions, bool)
            or not isinstance(active_positions, int)
            or active_positions <= 0
        ):
            raise ValueError("active_positions must be a positive integer")
        backend.capabilities.validate_runtime(
            max_batch_tokens=position_spec.max_batch_tokens,
            active_batches=active_positions,
        )

        self._receiver = receiver
        self._backend = backend
        self._instance_id = instance_id
        self._clock = clock
        self._positions: list[ActivePosition] = []
        try:
            for _ in range(active_positions):
                buffers = receiver.allocate_position_buffers(position_spec)
                self._positions.append(ActivePosition(position_spec, buffers))
        except BaseException:
            for position in self._positions:
                position.close()
            raise
        self._executor = ThreadPoolExecutor(
            max_workers=active_positions,
            thread_name_prefix="expertkit-worker-execution",
        )
        self._tasks: tuple[asyncio.Task[None], ...] = ()
        self._fatal_error: BackendFatalError | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def fixed_device_bytes(self) -> int:
        """Return device bytes reserved by all active input and output positions."""

        return sum(position.device_bytes for position in self._positions)

    @property
    def fixed_host_staging_bytes(self) -> int:
        """Return Host staging bytes reserved by all active positions."""

        return sum(position.host_staging_bytes for position in self._positions)

    async def start(self) -> None:
        """Start exactly one receiver loop for each active computation position."""

        if self._closed:
            raise RuntimeError("Worker execution is closed")
        if self._tasks:
            return
        self._tasks = tuple(
            asyncio.create_task(
                self._serve_position(position),
                name=f"worker-active-position-{index}",
            )
            for index, position in enumerate(self._positions)
        )

    async def wait(self) -> None:
        """Wait for all position loops and re-raise the first fatal Backend error."""

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
        for position in self._positions:
            position.close()

    async def _serve_position(self, position: ActivePosition) -> None:
        while True:
            try:
                received = await self._receiver.take()
            except ReceiverClosed:
                return
            fatal = await self._process(received, position)
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
        received: ReceivedWorkerBatch,
        position: ActivePosition,
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
                return None
            result = await self._run_position(position, received)
        except BackendRequestError as error:
            rejection = _request_error(error)
            logger.warning(
                "worker_computation_rejected",
                error_code=rejection.code.value,
                unavailable_expert_ids=rejection.unavailable_expert_ids,
            )
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
                await received.reject(result.rejection)
            elif result.output is None:
                raise RuntimeError("active position produced neither output nor rejection")
            else:
                await received.complete(result.output)
        except Exception:
            logger.error("worker_computation_response_failed", exc_info=True)
        finally:
            result.release()
        return None

    async def _run_position(
        self,
        position: ActivePosition,
        received: ReceivedWorkerBatch,
    ) -> PositionResult:
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(
            self._executor,
            partial(
                position.execute,
                received,
                self._backend,
                clock=self._clock,
            ),
        )
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
