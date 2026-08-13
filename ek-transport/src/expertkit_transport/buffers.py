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


_QUARANTINED_OUTPUT_SLOTS: list[_TensorSlot] = []
_QUARANTINED_OUTPUT_GRAPHS: list[tuple[object, ...]] = []


class OutputLease:
    """Hold one checked-out Tensor until routing has finished consuming it."""

    def __init__(self, slot: _TensorSlot) -> None:
        self._slot = slot
        self._consumed = False
        self._quarantined = False
        self._consumption_owners: list[object] = []

    @property
    def tensor(self) -> torch.Tensor:
        """Return the maximum-size Tensor owned by this lease."""

        return self._slot.tensor

    def mark_consumed(self, *, ownership_graph: tuple[object, ...] = ()) -> None:
        """Record a read and retain its operands until reuse is proven safe."""

        if self._consumed:
            raise RuntimeError("output Tensor consumption was already recorded")
        self._consumed = True
        self._consumption_owners.extend(ownership_graph)

    def retain_consumption_owners(self, *owners: object) -> None:
        """Add operands created after consumption tracking began."""

        if not self._consumed:
            raise RuntimeError("output Tensor consumption has not been recorded")
        self._consumption_owners.extend(owners)

    def quarantine(self) -> None:
        """Permanently withhold an output whose write completion is unprovable."""

        self._quarantined = True


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
        unsafe_error = (
            exc if isinstance(exc, TransportError) and exc.unsafe_tensor_ownership else None
        )
        if unsafe_error is not None and unsafe_error.unsafe_output:
            self._lease.quarantine()
        consumption_error = exc if exc is not None and self._lease._consumed else None
        cleanup = asyncio.create_task(
            self._pool._return(
                self._lease,
                wait_for_completion=consumption_error is not None,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        # A late ownership fatal must remain visible to routing/Blocking callers.
        # Cancellation cannot make the caller-owned Tensor safe to reuse.
        if cancellation is not None and unsafe_error is None and consumption_error is None:
            raise cancellation


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
        self._quarantined: list[_TensorSlot] = []
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
                self._quarantine_slot(slot)
                self._leased -= 1
                self._failed = True
                self._condition.notify_all()
            raise
        return OutputLease(slot)

    async def _return(
        self,
        lease: OutputLease,
        *,
        wait_for_completion: bool = False,
    ) -> None:
        completion_error: BaseException | None = None
        slot = lease._slot
        if lease._quarantined:
            async with self._condition:
                self._quarantine_slot(slot)
                self._leased -= 1
                self._failed = True
                self._condition.notify_all()
            return
        if lease._consumed and self._accelerator is not None:
            try:
                if slot.reuse_event is None:
                    slot.reuse_event = self._accelerator.create_event()
                slot.reuse_event.record(self._accelerator.current_stream())
            except BaseException as error:
                completion_error = error

        if completion_error is None and wait_for_completion and slot.reuse_event is not None:
            try:
                await self._synchronize_event_terminal(slot.reuse_event)
            except BaseException as error:
                completion_error = error

        async with self._condition:
            if completion_error is None:
                self._available.append(slot)
            else:
                self._quarantine_slot(slot)
            self._leased -= 1
            if completion_error is not None:
                self._failed = True
            self._condition.notify_all()
        if completion_error is not None:
            raise self._unsafe_consumption_error(lease) from completion_error

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
                self._quarantine_slot(slot)
                if release_error is None:
                    release_error = error

        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        if release_error is not None:
            raise release_error

    def _quarantine_slot(self, slot: _TensorSlot) -> None:
        if all(candidate is not slot for candidate in self._quarantined):
            self._quarantined.append(slot)
        if all(candidate is not slot for candidate in _QUARANTINED_OUTPUT_SLOTS):
            _QUARANTINED_OUTPUT_SLOTS.append(slot)

    async def _synchronize_event_terminal(self, event: torch.cuda.Event) -> None:
        loop = asyncio.get_running_loop()
        terminal = loop.run_in_executor(None, event.synchronize)
        while True:
            try:
                await asyncio.shield(terminal)
                return
            except asyncio.CancelledError:
                # The event owns routing operands. Repeated cancellation may
                # not release them before CUDA reaches the recorded terminal.
                continue

    def _unsafe_consumption_error(
        self,
        lease: OutputLease,
    ) -> TransportError:
        slot = lease._slot
        graph = (self, lease, slot, *lease._consumption_owners)
        _QUARANTINED_OUTPUT_GRAPHS.append(graph)
        return TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=False,
            unsafe_tensor_ownership=True,
            unsafe_output=True,
            diagnostic=(
                "CUDA output-consumption completion could not be proven; "
                "the routing ownership graph is retained and this process must restart"
            ),
        )
