"""Registration and reconnect supervision for both Controller streams."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import grpc
import structlog

from expertkit_worker.control.lifecycle import (
    ControllerConnection,
    HeartbeatSender,
    WorkerRegistration,
)
from expertkit_worker.control.weight_stream import WeightControlSession

logger = structlog.get_logger(__name__)

_RETRYABLE_CODES = {
    grpc.StatusCode.CANCELLED,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.UNAVAILABLE,
}


class ControllerSupervisor:
    """Register and reopen independent control streams on transient failures."""

    def __init__(
        self,
        *,
        connection: ControllerConnection,
        registration: WorkerRegistration,
        heartbeat: HeartbeatSender,
        weights: WeightControlSession,
        registration_timeout_secs: float,
        retry_initial_secs: float = 0.25,
        retry_max_secs: float = 5.0,
        stable_stream_secs: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("registration_timeout_secs", registration_timeout_secs),
            ("retry_initial_secs", retry_initial_secs),
            ("retry_max_secs", retry_max_secs),
            ("stable_stream_secs", stable_stream_secs),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if retry_max_secs < retry_initial_secs:
            raise ValueError("retry_max_secs cannot be smaller than retry_initial_secs")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._connection = connection
        self._registration = registration
        self._heartbeat = heartbeat
        self._weights = weights
        self._registration_timeout_secs = float(registration_timeout_secs)
        self._retry_initial_secs = float(retry_initial_secs)
        self._retry_max_secs = float(retry_max_secs)
        self._stable_stream_secs = float(stable_stream_secs)
        self._clock = clock
        self._retry_now = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._closing = False

    async def run(self) -> None:
        """Run registration and both streams until closed or a fatal error occurs."""

        async with self._run_lock:
            if self._closing:
                return
            await self._connection.start()
            retry_delay = self._retry_initial_secs
            while not self._closing:
                stream_started: float | None = None
                try:
                    result = await self._connection.register(
                        self._registration,
                        timeout_secs=self._registration_timeout_secs,
                    )
                    logger.info(
                        "worker_registered",
                        worker_id=self._registration.worker_id,
                        start_id=self._registration.start_id,
                        topology_version=result.topology_version,
                        placement_generation=result.placement_generation,
                    )
                    stream_started = self._clock()
                    await self._run_streams_once()
                    if not self._closing:
                        logger.warning("controller_stream_closed")
                except asyncio.CancelledError:
                    raise
                except grpc.aio.AioRpcError as error:
                    if self._closing:
                        return
                    if error.code() not in _RETRYABLE_CODES:
                        raise
                    logger.warning(
                        "controller_connection_retry",
                        error_code=error.code().name,
                        diagnostic=error.details(),
                        retry_delay_secs=retry_delay,
                    )

                if self._closing:
                    return
                if (
                    stream_started is not None
                    and self._clock() - stream_started >= self._stable_stream_secs
                ):
                    retry_delay = self._retry_initial_secs
                await self._wait_before_retry(retry_delay)
                retry_delay = min(self._retry_max_secs, retry_delay * 2)

    async def begin_shutdown(self) -> bool:
        """Notify Controller immediately and stop new expert-loading work."""

        heartbeat_changed = self._heartbeat.set_shutting_down()
        weights_changed = await self._weights.begin_shutdown()
        self._retry_now.set()
        return heartbeat_changed or weights_changed

    async def wait_shutdown_completion_sent(self) -> None:
        """Wait until authorized whole-Worker drain completion is handed to gRPC."""

        await self._weights.wait_shutdown_completion_sent()

    @property
    def shutdown_deadline(self) -> float | None:
        """Return the fixed deadline established by ``begin_shutdown``."""

        return self._weights.shutdown_deadline

    async def close(self) -> None:
        """End reconnects, control streams, and the shared channel idempotently."""

        if self._closing:
            return
        self._closing = True
        self._retry_now.set()
        self._heartbeat.close()
        await self._weights.close()
        await self._connection.close()

    async def _run_streams_once(self) -> None:
        tasks = (
            asyncio.create_task(
                self._heartbeat.run_once(self._connection),
                name="controller-heartbeat-stream",
            ),
            asyncio.create_task(
                self._weights.run_once(self._connection),
                name="controller-weight-stream",
            ),
        )
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _wait_before_retry(self, delay: float) -> None:
        self._retry_now.clear()
        if self._closing:
            return
        try:
            async with asyncio.timeout(delay):
                await self._retry_now.wait()
        except TimeoutError:
            pass
