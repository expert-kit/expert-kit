"""Bounded lifecycle for reusable Frontend partial-output buffers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import TracebackType

from expertkit_transport.buffers.base import OutputBufferProvider, OutputSpec, PreparedOutput
from expertkit_transport.errors import TransportError, TransportErrorCode


def _deadline_error() -> TransportError:
    return TransportError(
        TransportErrorCode.DEADLINE_EXCEEDED,
        retryable=False,
        diagnostic="deadline expired while waiting for an output buffer",
    )


class OutputLease:
    """Expose one checked-out output and record whether it was consumed."""

    def __init__(self, output: PreparedOutput) -> None:
        self.output = output
        self._consumed = False

    def mark_consumed(self) -> None:
        """Record that downstream Tensor work has read this output."""

        if self._consumed:
            raise RuntimeError("output consumption was already recorded")
        self._consumed = True


class _LeaseContext:
    def __init__(self, pool: OutputPool, monotonic_deadline: float) -> None:
        self._pool = pool
        self._deadline = monotonic_deadline
        self._lease: OutputLease | None = None

    async def __aenter__(self) -> OutputLease:
        self._lease = await self._pool._acquire(self._deadline)
        return self._lease

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lease is None:
            return
        cleanup = asyncio.create_task(self._pool._return(self._lease))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise


class OutputPool:
    """Preallocate a fixed number of outputs and reuse them safely.

    Allocation occurs in the constructor so callers can build pools while
    installing Topology rather than on the request hot path. Create and use a
    pool on one event loop.
    """

    def __init__(
        self,
        provider: OutputBufferProvider,
        spec: OutputSpec,
        capacity: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be positive")
        self.provider = provider
        self.spec = spec
        self.capacity = capacity
        self._clock = clock
        self._condition = asyncio.Condition()
        self._available: list[PreparedOutput] = []
        self._leased = 0
        self._failed = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

        try:
            for _ in range(capacity):
                output = provider.prepare(spec)
                self._available.append(output)
                provider.validate(output, spec)
        except BaseException:
            for output in self._available:
                provider.release(output)
            raise

    def lease(self, *, monotonic_deadline: float) -> _LeaseContext:
        """Return an async context that waits for one reusable output."""

        return _LeaseContext(self, monotonic_deadline)

    async def _acquire(self, monotonic_deadline: float) -> OutputLease:
        async with self._condition:
            if monotonic_deadline - self._clock() <= 0:
                raise _deadline_error()
            while not self._available:
                if self._closing or self._failed:
                    raise TransportError(
                        TransportErrorCode.UNAVAILABLE,
                        retryable=True,
                        diagnostic="output pool is unavailable",
                    )
                remaining = monotonic_deadline - self._clock()
                if remaining <= 0:
                    raise _deadline_error()
                try:
                    async with asyncio.timeout(remaining):
                        await self._condition.wait()
                except TimeoutError as error:
                    raise _deadline_error() from error

            if self._closing or self._failed:
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=True,
                    diagnostic="output pool is unavailable",
                )
            output = self._available.pop()
            self._leased += 1
        return OutputLease(output)

    async def _return(self, lease: OutputLease) -> None:
        completion_error: BaseException | None = None
        if lease._consumed:
            try:
                self.provider.after_consume(lease.output)
            except BaseException as error:
                completion_error = error

        async with self._condition:
            self._available.append(lease.output)
            self._leased -= 1
            if completion_error is not None:
                self._failed = True
            self._condition.notify_all()
        if completion_error is not None:
            raise completion_error

    async def close(self) -> None:
        """Stop new leases, drain active leases, and release every output."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closing = True
            self._condition.notify_all()
            await self._condition.wait_for(lambda: self._leased == 0)
            outputs = tuple(self._available)
            self._available.clear()

        release_error: BaseException | None = None
        for output in outputs:
            try:
                self.provider.release(output)
            except BaseException as error:
                if release_error is None:
                    release_error = error

        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        if release_error is not None:
            raise release_error
