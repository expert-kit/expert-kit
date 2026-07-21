"""Tests for the shared Worker receiver waiting and drain policy."""

import asyncio
import math

import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import ReceivedWorkerBatch, ReceiverClosed
from expertkit_transport.transports.queue import ReceiverQueue


def _batch(*, layer_id: int = 2, experts: tuple[int, ...] = (1, 3)) -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=layer_id,
        topology_version=11,
        hidden_states=torch.ones((1, 3), dtype=torch.float32),
        token_indices=None,
        expert_ids=torch.tensor([[experts[0]]], dtype=torch.int32),
        routing_weights=torch.ones((1, 1), dtype=torch.float32),
        distinct_expert_ids=experts,
    )


class _Batch(ReceivedWorkerBatch):
    def __init__(self, batch: WorkerBatch) -> None:
        self._batch = batch

    @property
    def trace_context(self):  # type: ignore[no-untyped-def]
        return None

    @property
    def batch(self) -> WorkerBatch:
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return math.inf

    @property
    def cancelled(self) -> bool:
        return False

    @property
    def output_destination(self) -> torch.Tensor | None:
        return None

    def release_input(self) -> None:
        pass

    async def complete(self, partial_output: torch.Tensor) -> None:
        del partial_output

    async def reject(self, error: TransportError) -> None:
        del error


def test_queue_bounds_waiting_batches_and_retained_bytes() -> None:
    async def scenario() -> None:
        pending: list[int] = []
        queue = ReceiverQueue(
            max_pending_batches=2,
            max_retained_bytes=10,
            on_pending_changed=pending.append,
        )
        first = _Batch(_batch(experts=(1,)))
        second = _Batch(_batch(experts=(2,)))
        third = _Batch(_batch(experts=(3,)))

        assert await queue.admit(first, retained_bytes=6) is None
        error = await queue.admit(second, retained_bytes=5)
        assert error is not None and error.code is TransportErrorCode.BUSY
        assert await queue.admit(second, retained_bytes=4) is None
        error = await queue.admit(third, retained_bytes=0)
        assert error is not None and error.code is TransportErrorCode.BUSY
        assert queue.pending_count == 2
        assert queue.retained_bytes == 10
        assert pending == [1, 2]

        assert await queue.take() is first
        assert queue.pending_count == 1
        assert queue.retained_bytes == 4
        assert queue.active_count == 1
        await queue.finish(first)
        assert queue.active_count == 0

    asyncio.run(scenario())


def test_queue_tracks_experts_across_waiting_and_active_work() -> None:
    async def scenario() -> None:
        queue = ReceiverQueue(max_pending_batches=2, max_retained_bytes=0)
        item = _Batch(_batch(experts=(1, 3)))
        assert await queue.admit(item, retained_bytes=0) is None
        assert queue.admitted_count(2, 1) == 1
        assert queue.admitted_count(2, 3) == 1

        active = await queue.take()
        with pytest.raises(TimeoutError):
            await queue.wait_experts_idle(
                ((2, 1),),
                monotonic_deadline=0.0,
            )
        await queue.finish(active)
        await queue.wait_experts_idle(((2, 1),), monotonic_deadline=math.inf)
        await queue.wait_all_idle(monotonic_deadline=math.inf)
        assert queue.admitted_count(2, 1) == 0

    asyncio.run(scenario())


def test_queue_cancels_only_waiting_work() -> None:
    async def scenario() -> None:
        queue = ReceiverQueue(max_pending_batches=2, max_retained_bytes=8)
        waiting = _Batch(_batch(experts=(1,)))
        active = _Batch(_batch(experts=(2,)))
        assert await queue.admit(active, retained_bytes=4) is None
        assert await queue.admit(waiting, retained_bytes=4) is None
        assert await queue.take() is active

        assert not await queue.cancel_waiting(active)
        assert await queue.cancel_waiting(waiting)
        assert not await queue.cancel_waiting(waiting)
        assert queue.pending_count == 0
        assert queue.retained_bytes == 0
        await queue.finish(active)

    asyncio.run(scenario())


def test_queue_rejects_draining_routes_until_cleared() -> None:
    async def scenario() -> None:
        queue = ReceiverQueue(max_pending_batches=1, max_retained_bytes=0)
        await queue.begin_drain(((2, 3),), min_topology_version=17, stop_all=False)

        error = await queue.admit(_Batch(_batch(experts=(1, 3))), retained_bytes=0)
        assert error is not None
        assert error.code is TransportErrorCode.DRAINING
        assert error.retryable
        assert error.min_topology_version == 17
        assert error.diagnostic == "Worker is draining the requested expert route"

        await queue.clear_expert_drains(((2, 3),))
        item = _Batch(_batch(experts=(1, 3)))
        assert await queue.admit(item, retained_bytes=0) is None
        await queue.cancel_waiting(item)

        await queue.begin_drain((), min_topology_version=19, stop_all=True)
        error = await queue.admit(_Batch(_batch(experts=(4,))), retained_bytes=0)
        assert error is not None
        assert error.code is TransportErrorCode.DRAINING
        assert error.min_topology_version == 19

    asyncio.run(scenario())


def test_queue_close_returns_waiting_and_unblocks_takers() -> None:
    async def scenario() -> None:
        queue = ReceiverQueue(max_pending_batches=2, max_retained_bytes=0)
        active = _Batch(_batch(experts=(1,)))
        waiting = _Batch(_batch(experts=(2,)))
        assert await queue.admit(active, retained_bytes=0) is None
        assert await queue.admit(waiting, retained_bytes=0) is None
        assert await queue.take() is active

        discarded = await queue.begin_close()
        assert discarded == (waiting,)
        assert queue.pending_count == 0
        assert queue.admitted_count(2, 2) == 0
        with pytest.raises(ReceiverClosed):
            await queue.take()

        close_wait = asyncio.create_task(queue.wait_active_empty())
        await asyncio.sleep(0)
        assert not close_wait.done()
        await queue.finish(active)
        await close_wait

        error = await queue.admit(_Batch(_batch(experts=(3,))), retained_bytes=0)
        assert error is not None and error.code is TransportErrorCode.UNAVAILABLE

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method", "args", "match"),
    [
        ("begin_drain", (((-1, 1),),), "unsigned 32-bit"),
        ("clear_expert_drains", (((1, 2), (1, 2)),), "duplicates"),
    ],
)
def test_queue_rejects_invalid_drain_experts(
    method: str,
    args: tuple[object, ...],
    match: str,
) -> None:
    async def scenario() -> None:
        queue = ReceiverQueue(max_pending_batches=1, max_retained_bytes=0)
        function = getattr(queue, method)
        kwargs = {"min_topology_version": 1, "stop_all": False} if method == "begin_drain" else {}
        with pytest.raises(ValueError, match=match):
            await function(*args, **kwargs)

    asyncio.run(scenario())
