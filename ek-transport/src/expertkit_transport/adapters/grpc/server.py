"""Worker-side gRPC receiver with one bounded pending area."""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from enum import StrEnum
from functools import partial
from typing import Any

import grpc
import torch

from expertkit_transport.adapters.grpc.codec import (
    GrpcProtocolError,
    decode_request,
    encode_error_response,
    encode_success_response,
)
from expertkit_transport.adapters.grpc.spec import (
    GrpcBatchSpec,
    calculate_message_limits,
)
from expertkit_transport.adapters.grpc.worker_buffers import GrpcWorkerPositionBuffers
from expertkit_transport.contracts import (
    ReceivedWorkerBatch,
    ReceiverClosed,
    TransportError,
    TransportErrorCode,
    WorkerBatch,
    WorkerBatchReceiver,
    WorkerPositionBuffers,
    WorkerPositionSpec,
)

_EXECUTE_METHOD_NAME = "Execute"
_SERVICE_NAME = "ek.worker.v2.ComputationService"
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_NATIVE_ERROR_STATUS = {
    TransportErrorCode.DEADLINE_EXCEEDED: grpc.StatusCode.DEADLINE_EXCEEDED,
    TransportErrorCode.CANCELLED: grpc.StatusCode.CANCELLED,
    TransportErrorCode.UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    TransportErrorCode.PROTOCOL: grpc.StatusCode.INVALID_ARGUMENT,
}


def _identity(payload: bytes) -> bytes:
    return payload


def _validate_drain_experts(
    experts: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    resolved = tuple(experts)
    seen: set[tuple[int, int]] = set()
    for key in resolved:
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("each draining expert must contain layer and expert IDs")
        for name, value in zip(("layer_id", "expert_id"), key, strict=True):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _UINT32_MAX
            ):
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        if key in seen:
            raise ValueError("draining experts must not contain duplicates")
        seen.add(key)
    return resolved


class _CallState(StrEnum):
    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"


class _NativeCallError(RuntimeError):
    """Carry a standard gRPC status from execution back to the RPC handler."""

    def __init__(self, status: grpc.StatusCode, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.status = status
        self.diagnostic = diagnostic


class _GrpcWorkItem(ReceivedWorkerBatch):
    def __init__(
        self,
        owner: GrpcWorkerServer,
        batch: WorkerBatch,
        monotonic_deadline: float,
        retained_bytes: int,
    ) -> None:
        self._owner = owner
        self._batch: WorkerBatch | None = batch
        self.layer_id = batch.layer_id
        self.token_count = batch.token_count
        self.distinct_expert_ids = batch.distinct_expert_ids
        self._deadline = monotonic_deadline
        self.retained_bytes = retained_bytes
        self.state = _CallState.WAITING
        self._cancelled = False
        self.response: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("received gRPC input has already been released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def release_input(self) -> None:
        """Drop decoded protobuf Tensor views after the active-position copy."""

        self._owner._require_active(self)
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        """Serialize a success unless the caller already discarded the response."""

        await self._owner._complete_success(self, partial_output)

    async def reject(self, error: TransportError) -> None:
        """Serialize one structured computation rejection."""

        await self._owner._complete_error(self, error)


class _PendingArea:
    def __init__(self, max_count: int, max_retained_bytes: int) -> None:
        self.max_count = max_count
        self.max_retained_bytes = max_retained_bytes
        self._items: deque[_GrpcWorkItem] = deque()
        self._retained_bytes = 0
        self._condition = asyncio.Condition()
        self._closing = False

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def retained_bytes(self) -> int:
        return self._retained_bytes

    async def try_admit(self, item: _GrpcWorkItem) -> bool:
        async with self._condition:
            if self._closing:
                return False
            if len(self._items) >= self.max_count:
                return False
            if self._retained_bytes + item.retained_bytes > self.max_retained_bytes:
                return False
            self._items.append(item)
            self._retained_bytes += item.retained_bytes
            self._condition.notify_all()
            return True

    async def take(self) -> _GrpcWorkItem:
        async with self._condition:
            await self._condition.wait_for(lambda: self._items or self._closing)
            if not self._items:
                raise ReceiverClosed("gRPC receiver is closed")
            item = self._items.popleft()
            self._retained_bytes -= item.retained_bytes
            item.state = _CallState.ACTIVE
            self._condition.notify_all()
            return item

    async def cancel_waiting(self, item: _GrpcWorkItem) -> bool:
        async with self._condition:
            if item.state is not _CallState.WAITING:
                return False
            try:
                self._items.remove(item)
            except ValueError:
                return False
            self._retained_bytes -= item.retained_bytes
            item.state = _CallState.FINISHED
            self._condition.notify_all()
            return True

    async def close(self) -> tuple[_GrpcWorkItem, ...]:
        async with self._condition:
            self._closing = True
            waiting = tuple(self._items)
            self._items.clear()
            self._retained_bytes = 0
            for item in waiting:
                item.state = _CallState.FINISHED
            self._condition.notify_all()
            return waiting

    async def wait_for_count(self, expected: int) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: len(self._items) == expected or self._closing)


class GrpcWorkerServer(WorkerBatchReceiver):
    """Receive plaintext unary calls and expose one Transport-owned waiting area."""

    def __init__(
        self,
        listen: str,
        batch_spec: GrpcBatchSpec,
        *,
        max_active_batches: int,
        max_pending_batches: int,
        cpu_workers: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not listen:
            raise ValueError("listen must not be empty")
        for name, value in (
            ("max_active_batches", max_active_batches),
            ("max_pending_batches", max_pending_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        resolved_cpu_workers = cpu_workers if cpu_workers is not None else max_active_batches
        if (
            isinstance(resolved_cpu_workers, bool)
            or not isinstance(resolved_cpu_workers, int)
            or resolved_cpu_workers <= 0
        ):
            raise ValueError("cpu_workers must be a positive integer")

        self._listen = listen
        self._spec = batch_spec
        self._limits = calculate_message_limits(batch_spec)
        self._maximum_concurrent_rpcs = max_active_batches + max_pending_batches
        self._pending = _PendingArea(
            max_pending_batches,
            max_pending_batches * self._limits.request_bytes,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=resolved_cpu_workers,
            thread_name_prefix="expertkit-grpc-server",
        )
        self._clock = clock
        self._server: grpc.aio.Server | None = None
        self._start_lock = asyncio.Lock()
        self._admitted_experts: Counter[tuple[int, int]] = Counter()
        self._draining_experts: dict[tuple[int, int], int] = {}
        self._stop_all_min_topology_version: int | None = None
        self._active_count = 0
        self._state_condition = asyncio.Condition()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._bound_port: int | None = None

    @property
    def bound_port(self) -> int:
        """Return the bound TCP port after startup."""

        if self._bound_port is None:
            raise RuntimeError("gRPC receiver has not been started")
        return self._bound_port

    @property
    def pending_count(self) -> int:
        """Return the number of decoded calls waiting for Worker execution."""

        return self._pending.count

    @property
    def pending_retained_bytes(self) -> int:
        """Return raw Tensor bytes retained by waiting calls."""

        return self._pending.retained_bytes

    @property
    def active_count(self) -> int:
        """Return batches already taken by Worker execution."""

        return self._active_count

    async def start(self) -> None:
        """Start the finite plaintext `grpc.aio` computation server."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("gRPC receiver is closing")
            if self._server is not None:
                return
            server = grpc.aio.server(
                options=self._limits.server_options,
                maximum_concurrent_rpcs=self._maximum_concurrent_rpcs,
            )
            method = grpc.unary_unary_rpc_method_handler(
                self._execute,
                request_deserializer=_identity,
                response_serializer=_identity,
            )
            service = grpc.method_handlers_generic_handler(
                _SERVICE_NAME,
                {_EXECUTE_METHOD_NAME: method},
            )
            server.add_generic_rpc_handlers((service,))
            port = server.add_insecure_port(self._listen)
            if port == 0:
                raise RuntimeError("gRPC receiver could not bind its listen address")
            await server.start()
            self._server = server
            self._bound_port = port

    async def take(self) -> ReceivedWorkerBatch:
        """Move one waiting batch directly into Worker execution ownership."""

        item = await self._pending.take()
        async with self._state_condition:
            self._active_count += 1
            self._state_condition.notify_all()
        return item

    def allocate_position_buffers(self, spec: WorkerPositionSpec) -> WorkerPositionBuffers:
        """Allocate fixed gRPC staging for one Worker active position."""

        expected = (
            self._spec.max_batch_tokens,
            self._spec.hidden_dim,
            self._spec.top_k,
            self._spec.dtype,
        )
        actual = (spec.max_batch_tokens, spec.hidden_dim, spec.top_k, spec.dtype)
        if actual != expected:
            raise ValueError("Worker position shape does not match the gRPC endpoint")
        return GrpcWorkerPositionBuffers(spec)

    async def close(self) -> None:
        """Stop RPC admission and release waiting calls and CPU workers."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        """Reject new matching calls while preserving already admitted work."""

        selected = _validate_drain_experts(experts)
        if (
            isinstance(min_topology_version, bool)
            or not isinstance(min_topology_version, int)
            or not 0 < min_topology_version <= _UINT64_MAX
        ):
            raise ValueError("min_topology_version must be a positive uint64")
        if not isinstance(stop_all, bool):
            raise ValueError("stop_all must be a Boolean")
        if not selected and not stop_all:
            raise ValueError("an expert drain must name at least one expert")

        async with self._state_condition:
            for key in selected:
                current = self._draining_experts.get(key, 0)
                self._draining_experts[key] = max(current, min_topology_version)
            if stop_all:
                current = self._stop_all_min_topology_version or 0
                self._stop_all_min_topology_version = max(current, min_topology_version)
            self._state_condition.notify_all()

    async def clear_expert_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        """Clear per-expert gates after later assignments become ready."""

        selected = _validate_drain_experts(experts)
        async with self._state_condition:
            for key in selected:
                self._draining_experts.pop(key, None)
            self._state_condition.notify_all()

    async def wait_experts_idle(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        monotonic_deadline: float,
    ) -> None:
        """Wait until no waiting or active batch uses the selected experts."""

        selected = _validate_drain_experts(experts)

        def idle() -> bool:
            return all(self._admitted_experts[key] == 0 for key in selected)

        async with self._state_condition:
            while not idle():
                remaining = monotonic_deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError("deadline expired while waiting for admitted expert use")
                if math.isinf(remaining):
                    await self._state_condition.wait()
                else:
                    async with asyncio.timeout(remaining):
                        await self._state_condition.wait()

    def admitted_count(self, layer_id: int, expert_id: int) -> int:
        """Return waiting plus active batches that name one expert."""

        return self._admitted_experts[(layer_id, expert_id)]

    async def _execute(
        self,
        payload: bytes,
        context: grpc.aio.ServicerContext,
    ) -> bytes:
        if self._closing:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "Worker is shutting down")
        try:
            batch = await self._run_cpu(decode_request, payload, self._spec)
        except GrpcProtocolError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            raise AssertionError("context.abort must terminate the handler") from error

        remaining = context.time_remaining()
        deadline = math.inf if remaining is None else self._clock() + max(0.0, remaining)
        item = _GrpcWorkItem(self, batch, deadline, len(payload))
        rejection = await self._admit(item)
        if rejection is not None:
            return await self._run_cpu(encode_error_response, rejection, self._spec)

        try:
            return await asyncio.shield(item.response)
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(self._cancel(item))
            with suppress(BaseException):
                await asyncio.shield(cleanup)
            raise
        except _NativeCallError as error:
            await context.abort(error.status, error.diagnostic)
            raise AssertionError("context.abort must terminate the handler") from error
        except Exception as error:
            await context.abort(grpc.StatusCode.INTERNAL, "failed to encode Worker response")
            raise AssertionError("context.abort must terminate the handler") from error

    async def _admit(self, item: _GrpcWorkItem) -> TransportError | None:
        async with self._state_condition:
            draining = self._drain_error_locked(item)
            if draining is not None:
                return draining
            admitted = await self._pending.try_admit(item)
            if not admitted:
                return TransportError(
                    TransportErrorCode.BUSY,
                    retryable=True,
                    diagnostic="Worker Transport waiting area is full",
                )
            for expert_id in item.distinct_expert_ids:
                self._admitted_experts[(item.layer_id, expert_id)] += 1
            self._state_condition.notify_all()
        return None

    def _drain_error_locked(self, item: _GrpcWorkItem) -> TransportError | None:
        min_topology_version = self._stop_all_min_topology_version
        for expert_id in item.distinct_expert_ids:
            expert_version = self._draining_experts.get((item.layer_id, expert_id))
            if expert_version is not None:
                min_topology_version = max(min_topology_version or 0, expert_version)
        if min_topology_version is None:
            return None
        return TransportError(
            TransportErrorCode.DRAINING,
            retryable=True,
            min_topology_version=min_topology_version,
            diagnostic="Worker is draining the requested expert route",
        )

    async def _cancel(self, item: _GrpcWorkItem) -> None:
        async with self._state_condition:
            item._cancelled = True
            self._state_condition.notify_all()
        if await self._pending.cancel_waiting(item):
            await self._release_admitted(item, was_active=False)

    async def _wait_pending_count(self, expected: int) -> None:
        await self._pending.wait_for_count(expected)

    async def _wait_cancelled(self, item: _GrpcWorkItem) -> None:
        async with self._state_condition:
            await self._state_condition.wait_for(lambda: item.cancelled)

    async def _complete_success(
        self,
        item: _GrpcWorkItem,
        partial_output: torch.Tensor,
    ) -> None:
        self._require_active(item)
        try:
            if not item.cancelled:
                if partial_output.ndim != 2 or partial_output.shape[0] != item.token_count:
                    raise ValueError("partial output token count does not match the received batch")
                payload = await self._run_cpu(
                    encode_success_response,
                    partial_output,
                    self._spec,
                )
                if not item.cancelled and not item.response.done():
                    item.response.set_result(payload)
        except BaseException as error:
            if not item.cancelled and not item.response.done():
                item.response.set_exception(error)
            raise
        finally:
            await self._finish_active(item)

    async def _complete_error(
        self,
        item: _GrpcWorkItem,
        error: TransportError,
    ) -> None:
        self._require_active(item)
        try:
            if not item.cancelled:
                status = _NATIVE_ERROR_STATUS.get(error.code)
                if status is None:
                    payload = await self._run_cpu(encode_error_response, error, self._spec)
                    if not item.cancelled and not item.response.done():
                        item.response.set_result(payload)
                elif not item.response.done():
                    item.response.set_exception(
                        _NativeCallError(status, error.diagnostic or error.code.value)
                    )
        except BaseException as cause:
            if not item.cancelled and not item.response.done():
                item.response.set_exception(cause)
            raise
        finally:
            await self._finish_active(item)

    async def _finish_active(self, item: _GrpcWorkItem) -> None:
        item._batch = None
        item.state = _CallState.FINISHED
        cleanup = asyncio.create_task(self._release_admitted(item, was_active=True))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _release_admitted(
        self,
        item: _GrpcWorkItem,
        *,
        was_active: bool,
    ) -> None:
        async with self._state_condition:
            if was_active:
                self._active_count -= 1
            for expert_id in item.distinct_expert_ids:
                key = (item.layer_id, expert_id)
                self._admitted_experts[key] -= 1
                if self._admitted_experts[key] == 0:
                    del self._admitted_experts[key]
            self._state_condition.notify_all()

    def _require_active(self, item: _GrpcWorkItem) -> None:
        if item.state is not _CallState.ACTIVE:
            raise RuntimeError("Worker batch completion requires one active received batch")

    async def _close(self) -> None:
        self._closing = True
        if self._server is not None:
            await self._server.stop(None)
        waiting = await self._pending.close()
        for item in waiting:
            item._cancelled = True
            await self._release_admitted(item, was_active=False)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._executor.shutdown, wait=True, cancel_futures=True),
        )

    async def _run_cpu(self, function: Callable[..., Any], *args: object) -> Any:
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(self._executor, function, *args)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await work
            raise
