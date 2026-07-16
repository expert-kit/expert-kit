"""Worker process startup, supervision, graceful shutdown, and cleanup."""

from __future__ import annotations

import asyncio
import signal
import time
from collections.abc import Callable
from typing import Protocol

import structlog

logger = structlog.get_logger(__name__)


class WorkerShutdownTimeout(TimeoutError):
    """Indicate that Controller-authorized shutdown exceeded its fixed deadline."""


class _AsyncStartClose(Protocol):
    async def start(self) -> None:
        """Start the component."""

    async def close(self) -> None:
        """Close the component."""


class _Execution(_AsyncStartClose, Protocol):
    async def wait(self) -> None:
        """Wait for execution loops to end."""


class _WeightManager(Protocol):
    def start(self) -> None:
        """Start background weight work."""

    async def wait_fatal(self) -> BaseException:
        """Wait for an unrecoverable placement failure."""

    async def close(self) -> None:
        """Close loading and ready weights."""


class _Control(Protocol):
    async def run(self) -> None:
        """Supervise Controller streams."""

    async def begin_shutdown(self) -> bool:
        """Send shutdown state and stop new loading."""

    async def wait_shutdown_completion_sent(self) -> None:
        """Wait for authorized drain completion to enter the stream."""

    @property
    def shutdown_deadline(self) -> float | None:
        """Return the overall shutdown deadline."""

    async def close(self) -> None:
        """Close Controller streams and connections."""


class _SyncClose(Protocol):
    def close(self) -> None:
        """Close the blocking resource."""


class WorkerApplication:
    """Own every process-lifetime component and their shutdown ordering."""

    def __init__(
        self,
        *,
        transfer: _AsyncStartClose,
        manager: _WeightManager,
        peer_server: _AsyncStartClose,
        computation_server: _AsyncStartClose,
        execution: _Execution,
        control: _Control,
        disk_cache: _SyncClose,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._transfer = transfer
        self._manager = manager
        self._peer_server = peer_server
        self._computation_server = computation_server
        self._execution = execution
        self._control = control
        self._disk_cache = disk_cache
        self._clock = clock
        self._shutdown_requested = asyncio.Event()
        self._started = asyncio.Event()
        self._control_task: asyncio.Task[None] | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._manager_fatal_task: asyncio.Task[BaseException] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def started(self) -> asyncio.Event:
        """Expose startup completion for launchers and integration tests."""

        return self._started

    def request_shutdown(self) -> bool:
        """Request graceful shutdown without performing work in a signal handler."""

        if self._shutdown_requested.is_set():
            return False
        self._shutdown_requested.set()
        return True

    async def run(self) -> None:
        """Start all services and supervise them until shutdown or fatal failure."""

        if self._running:
            raise RuntimeError("Worker application is already running")
        self._running = True
        shutdown_wait: asyncio.Task[bool] | None = None
        try:
            await self._start()
            shutdown_wait = asyncio.create_task(
                self._shutdown_requested.wait(),
                name="worker-shutdown-request",
            )
            monitors = self._monitor_tasks()
            done, _pending = await asyncio.wait(
                (*monitors, shutdown_wait),
                return_when=asyncio.FIRST_COMPLETED,
            )
            failed = next((task for task in monitors if task in done), None)
            if failed is not None:
                await self._raise_monitor_result(failed)
            await self._graceful_shutdown(monitors)
        finally:
            if shutdown_wait is not None:
                shutdown_wait.cancel()
                await asyncio.gather(shutdown_wait, return_exceptions=True)
            await self.close()
            self._running = False

    async def close(self) -> None:
        """Close every initialized component once in dependency order."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="worker-app-close")
        await asyncio.shield(self._close_task)

    def install_signal_handlers(self) -> Callable[[], None]:
        """Install SIGINT and SIGTERM handlers and return their remover."""

        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        for selected in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(selected, self.request_shutdown)
            except NotImplementedError:
                continue
            installed.append(selected)

        def remove() -> None:
            for selected in installed:
                loop.remove_signal_handler(selected)

        return remove

    async def _start(self) -> None:
        await self._transfer.start()
        self._manager.start()
        await self._peer_server.start()
        await self._computation_server.start()
        await self._execution.start()
        self._control_task = asyncio.create_task(
            self._control.run(),
            name="worker-controller-supervisor",
        )
        self._execution_task = asyncio.create_task(
            self._execution.wait(),
            name="worker-execution-supervisor",
        )
        self._manager_fatal_task = asyncio.create_task(
            self._manager.wait_fatal(),
            name="worker-weight-fatal-supervisor",
        )
        self._started.set()
        logger.info("worker_started")

    def _monitor_tasks(
        self,
    ) -> tuple[asyncio.Task[None], asyncio.Task[None], asyncio.Task[BaseException]]:
        if (
            self._control_task is None
            or self._execution_task is None
            or self._manager_fatal_task is None
        ):
            raise RuntimeError("Worker monitor tasks have not been started")
        return self._control_task, self._execution_task, self._manager_fatal_task

    async def _graceful_shutdown(
        self,
        monitors: tuple[asyncio.Task[None], asyncio.Task[None], asyncio.Task[BaseException]],
    ) -> None:
        await self._control.begin_shutdown()
        deadline = self._control.shutdown_deadline
        if deadline is None:
            raise RuntimeError("shutdown request did not establish a deadline")
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise WorkerShutdownTimeout("Worker shutdown deadline expired")

        completion = asyncio.create_task(
            self._control.wait_shutdown_completion_sent(),
            name="worker-shutdown-completion",
        )
        try:
            async with asyncio.timeout(remaining):
                done, _pending = await asyncio.wait(
                    (*monitors, completion),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                failed = next((task for task in monitors if task in done), None)
                if failed is not None:
                    await self._raise_monitor_result(failed)
                await completion
        except TimeoutError as error:
            raise WorkerShutdownTimeout("Worker shutdown deadline expired") from error
        finally:
            completion.cancel()
            await asyncio.gather(completion, return_exceptions=True)

    async def _raise_monitor_result(self, task: asyncio.Task[object]) -> None:
        if task is self._manager_fatal_task:
            error = await task
            raise error
        await task
        if task is self._control_task:
            raise RuntimeError("Controller supervisor stopped unexpectedly")
        raise RuntimeError("Worker execution stopped unexpectedly")

    async def _close(self) -> None:
        await self._control.close()
        await self._execution.close()
        await self._peer_server.close()
        await self._manager.close()
        await self._transfer.close()
        self._disk_cache.close()
        monitors = tuple(
            task
            for task in (
                self._control_task,
                self._execution_task,
                self._manager_fatal_task,
            )
            if task is not None
        )
        for task in monitors:
            task.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        logger.info("worker_stopped")
