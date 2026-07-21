"""Test-only in-process implementation of both Transport interfaces."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerBatchReceiver,
    WorkerEndpointConfig,
    WorkerTransport,
)
from expertkit_transport.transports.queue import ReceiverQueue
from expertkit_transport.transports.validation import (
    validate_received_routing,
    validate_worker_batch,
)


def _copy_batch(batch: WorkerBatch, config: WorkerEndpointConfig) -> WorkerBatch:
    validate_worker_batch(batch, config)
    if batch.token_indices is None:
        hidden_states = batch.hidden_states
    else:
        try:
            hidden_states = torch.index_select(batch.hidden_states, 0, batch.token_indices)
        except (IndexError, RuntimeError) as error:
            raise ValueError("Worker batch token indices are invalid") from error
    hidden_states = hidden_states.detach().to(device="cpu").contiguous().clone()
    expert_ids = batch.expert_ids.detach().to(device="cpu").contiguous().clone()
    routing_weights = batch.routing_weights.detach().to(device="cpu").contiguous().clone()
    distinct_expert_ids = validate_received_routing(
        expert_ids,
        routing_weights,
        config.experts_per_layer,
    )
    return WorkerBatch(
        instance_id=batch.instance_id,
        layer_id=batch.layer_id,
        topology_version=batch.topology_version,
        hidden_states=hidden_states,
        token_indices=None,
        expert_ids=expert_ids,
        routing_weights=routing_weights,
        distinct_expert_ids=distinct_expert_ids,
    )


def _validate_output(
    output: torch.Tensor,
    batch: WorkerBatch,
    config: WorkerEndpointConfig,
) -> None:
    if output.shape != (batch.token_count, config.hidden_dim):
        raise ValueError("output must have shape [token_count, hidden_dim]")
    if output.dtype != config.dtype:
        raise ValueError("output dtype does not match the endpoint")
    if output.device.type != "cpu" or batch.hidden_states.device.type != "cpu":
        raise ValueError("the in-memory test Transport supports CPU Tensors only")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")


class _InMemoryBatchBuffers(WorkerBatchBuffers):
    def __init__(self, _config: BatchBufferConfig) -> None:
        self._closed = False

    @property
    def host_staging_bytes(self) -> int:
        return 0

    def copy_input(
        self,
        batch: WorkerBatch,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        self._require_open()
        hidden_states.copy_(batch.hidden_states)
        expert_ids.copy_(batch.expert_ids)
        routing_weights.copy_(batch.routing_weights)

    def copy_output(
        self,
        partial_output: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        self._require_open()
        if destination is None:
            return partial_output
        destination.copy_(partial_output)
        return destination

    def close(self) -> None:
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("in-memory batch buffers are closed")


@dataclass(slots=True)
class _SharedState:
    config: WorkerEndpointConfig
    queue: ReceiverQueue
    receiver_started: bool = False
    receiver_closed: bool = False


class _InMemoryReceivedBatch(ReceivedBatch):
    def __init__(
        self,
        state: _SharedState,
        batch: WorkerBatch,
        output: torch.Tensor,
        monotonic_deadline: float,
    ) -> None:
        self._state = state
        self._batch: WorkerBatch | None = batch
        self._output = output
        self._deadline = monotonic_deadline
        self._cancelled = False
        self._response: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    @property
    def trace_context(self) -> None:
        return None

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("received in-memory input has already been released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def output_destination(self) -> None:
        return None

    def release_input(self) -> None:
        self._state.queue.require_active(self)
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        self._state.queue.require_active(self)
        if not self._cancelled:
            self._output.copy_(partial_output)
        await self._state.queue.finish(self)
        if not self._response.done():
            self._response.set_result(None)

    async def reject(self, error: TransportError) -> None:
        self._state.queue.require_active(self)
        await self._state.queue.finish(self)
        if not self._response.done():
            self._response.set_exception(error)

    def cancel(self) -> None:
        self._cancelled = True

    def discard_waiting(self, error: TransportError | None = None) -> None:
        self._batch = None
        if self._response.done():
            return
        if error is None:
            self._response.cancel()
        else:
            self._response.set_exception(error)


class InMemoryWorkerTransport(WorkerTransport):
    """Send copied CPU batches through one in-process receiver queue."""

    def __init__(self, state: _SharedState) -> None:
        self._state = state
        self._started = False
        self._closing = False

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("in-memory Transport is closing")
        self._started = True

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        if not self._started:
            raise RuntimeError("in-memory Transport has not been started")
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="in-memory Transport is closing",
            )
        _validate_output(output, batch, self._state.config)
        if monotonic_deadline <= time.monotonic():
            raise TransportError(
                TransportErrorCode.DEADLINE_EXCEEDED,
                retryable=False,
                diagnostic="deadline expired before in-memory admission",
            )
        try:
            copied = _copy_batch(batch, self._state.config)
        except ValueError as error:
            raise TransportError(
                TransportErrorCode.INVALID_REQUEST,
                retryable=False,
                diagnostic=str(error),
            ) from error
        received = _InMemoryReceivedBatch(
            self._state,
            copied,
            output,
            monotonic_deadline,
        )
        rejection = await self._state.queue.admit(received, retained_bytes=0)
        if rejection is not None:
            received.discard_waiting()
            raise rejection
        try:
            remaining = monotonic_deadline - time.monotonic()
            if math.isinf(remaining):
                await asyncio.shield(received._response)
            else:
                async with asyncio.timeout(max(0.0, remaining)):
                    await asyncio.shield(received._response)
        except asyncio.CancelledError:
            await self._cancel(received)
            raise
        except TimeoutError as error:
            await self._cancel(received)
            raise TransportError(
                TransportErrorCode.DEADLINE_EXCEEDED,
                retryable=False,
                diagnostic="in-memory request deadline expired",
            ) from error

    async def close(self) -> None:
        self._closing = True

    async def _cancel(self, received: _InMemoryReceivedBatch) -> None:
        received.cancel()
        if await self._state.queue.cancel_waiting(received):
            received.discard_waiting()


class InMemoryWorkerBatchReceiver(WorkerBatchReceiver):
    """Expose the Worker side of the test-only in-process Transport."""

    def __init__(self, state: _SharedState) -> None:
        self._state = state
        self._close_task: asyncio.Task[None] | None = None

    @property
    def pending_count(self) -> int:
        """Return waiting batches for deterministic conformance checks."""

        return self._state.queue.pending_count

    @property
    def active_count(self) -> int:
        """Return batches held by the simulated Worker execution path."""

        return self._state.queue.active_count

    async def start(self) -> None:
        if self._state.receiver_closed:
            raise RuntimeError("in-memory receiver is closed")
        self._state.receiver_started = True

    async def receive(self) -> ReceivedBatch:
        if not self._state.receiver_started:
            raise RuntimeError("in-memory receiver has not been started")
        return await self._state.queue.take()

    def create_batch_buffers(self, config: BatchBufferConfig) -> WorkerBatchBuffers:
        expected = (
            self._state.config.max_batch_tokens,
            self._state.config.hidden_dim,
            self._state.config.top_k,
            self._state.config.dtype,
            torch.device("cpu"),
        )
        actual = (
            config.max_batch_tokens,
            config.hidden_dim,
            config.top_k,
            config.dtype,
            config.device,
        )
        if actual != expected:
            raise ValueError("Worker buffer shape does not match the in-memory endpoint")
        return _InMemoryBatchBuffers(config)

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        await self._state.queue.begin_drain(
            experts,
            min_topology_version=min_topology_version,
            stop_all=stop_all,
        )

    async def clear_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        await self._state.queue.clear_expert_drains(experts)

    async def wait_idle(
        self,
        experts: Iterable[tuple[int, int]] | None,
        *,
        monotonic_deadline: float,
    ) -> None:
        if experts is None:
            await self._state.queue.wait_all_idle(monotonic_deadline=monotonic_deadline)
        else:
            await self._state.queue.wait_experts_idle(
                experts,
                monotonic_deadline=monotonic_deadline,
            )

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._state.receiver_closed:
            return
        self._state.receiver_closed = True
        waiting = await self._state.queue.begin_close()
        error = TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            diagnostic="in-memory receiver is closed",
        )
        for item in waiting:
            assert isinstance(item, _InMemoryReceivedBatch)
            item.discard_waiting(error)
        await self._state.queue.wait_active_empty()


def create_in_memory_pair(
    config: WorkerEndpointConfig,
    *,
    max_pending_batches: int,
) -> tuple[InMemoryWorkerBatchReceiver, InMemoryWorkerTransport]:
    """Return connected test-only implementations without touching production code."""

    state = _SharedState(
        config=config,
        queue=ReceiverQueue(
            max_pending_batches=max_pending_batches,
            max_retained_bytes=0,
        ),
    )
    return InMemoryWorkerBatchReceiver(state), InMemoryWorkerTransport(state)
