"""Bounded reuse of ordinary Frontend output Tensors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

import torch

from expertkit_transport._accelerator import TorchAccelerator, accelerator_for
from expertkit_transport.batches import ACTIVATION_DTYPES
from expertkit_transport.errors import TransportError, TransportErrorCode


def _deadline_error() -> TransportError:
    return TransportError(
        TransportErrorCode.DEADLINE_EXCEEDED,
        retryable=False,
        diagnostic="deadline expired while waiting for an output Tensor",
    )


@dataclass(slots=True)
class _TensorSlot:
    tensor: torch.Tensor
    reuse_event: torch.Event | None = None


class OutputLease:
    """Hold one checked-out Tensor until routing has finished consuming it."""

    def __init__(self, slot: _TensorSlot) -> None:
        self._slot = slot
        self._consumed = False

    @property
    def tensor(self) -> torch.Tensor:
        """Return the maximum-size Tensor owned by this lease."""

        return self._slot.tensor

    def mark_consumed(self) -> None:
        """Record that current-stream work has read this Tensor."""

        if self._consumed:
            raise RuntimeError("output Tensor consumption was already recorded")
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
    """Preallocate ordinary Tensors used for concurrent Worker partial results.

    Concrete Transports do not allocate or own these Tensors. They receive one
    checked-out Tensor as the destination for a single call and privately manage
    any Host staging, shared-memory slot, or device-transfer synchronization.
    """

    def __init__(
        self,
        *,
        max_batch_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("max_batch_tokens", max_batch_tokens),
            ("hidden_dim", hidden_dim),
            ("capacity", capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dtype not in ACTIVATION_DTYPES:
            raise ValueError("output dtype must be FP16, BF16, or FP32")

        self.max_batch_tokens = max_batch_tokens
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self._accelerator: TorchAccelerator | None = accelerator_for(self.device)
        self.capacity = capacity
        self._clock = clock
        self._condition = asyncio.Condition()
        self._available = [
            _TensorSlot(
                torch.empty(
                    (max_batch_tokens, hidden_dim),
                    dtype=dtype,
                    device=self.device,
                )
            )
            for _ in range(capacity)
        ]
        self._leased = 0
        self._failed = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def lease(self, *, monotonic_deadline: float) -> _LeaseContext:
        """Wait for one reusable output Tensor until the absolute deadline."""

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
                        diagnostic="output Tensor pool is unavailable",
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
                    diagnostic="output Tensor pool is unavailable",
                )
            slot = self._available.pop()
            self._leased += 1

        try:
            if slot.reuse_event is not None:
                assert self._accelerator is not None
                self._accelerator.current_stream().wait_event(slot.reuse_event)
        except BaseException:
            async with self._condition:
                self._available.append(slot)
                self._leased -= 1
                self._failed = True
                self._condition.notify_all()
            raise
        return OutputLease(slot)

    async def _return(self, lease: OutputLease) -> None:
        completion_error: BaseException | None = None
        slot = lease._slot
        if lease._consumed and self._accelerator is not None:
            try:
                if slot.reuse_event is None:
                    slot.reuse_event = self._accelerator.create_event()
                slot.reuse_event.record(self._accelerator.current_stream())
            except BaseException as error:
                completion_error = error

        async with self._condition:
            self._available.append(slot)
            self._leased -= 1
            if completion_error is not None:
                self._failed = True
            self._condition.notify_all()
        if completion_error is not None:
            raise completion_error

    async def close(self) -> None:
        """Stop new leases, drain active leases, and release all Tensor storage."""

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
            slots = tuple(self._available)
            self._available.clear()

        release_error: BaseException | None = None
        for slot in slots:
            if slot.reuse_event is None:
                continue
            try:
                slot.reuse_event.synchronize()
            except BaseException as error:
                if release_error is None:
                    release_error = error

        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        if release_error is not None:
            raise release_error
