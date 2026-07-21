"""Frontend shared-memory data path with gRPC control notifications."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from typing import Any

import grpc
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import (
    TransportError,
    TransportErrorCode,
    TransportProtocolError,
)
from expertkit_transport.transports.base import WorkerEndpointConfig, WorkerTransport
from expertkit_transport.transports.shm.buffers import (
    ShmTransferBufferPool,
    ShmTransferBuffers,
)
from expertkit_transport.transports.shm.codec import (
    SHM_CONTROL_MESSAGE_BYTES,
    ExecuteSlot,
    decode_close_response,
    decode_execute_response,
    decode_open_response,
    encode_close_request,
    encode_execute_request,
    encode_open_request,
)
from expertkit_transport.transports.validation import validate_worker_batch

_OPEN_METHOD = "/ek.worker.v2.ComputationService/OpenSharedMemory"
_EXECUTE_METHOD = "/ek.worker.v2.ComputationService/ExecuteSharedMemory"
_CLOSE_METHOD = "/ek.worker.v2.ComputationService/CloseSharedMemory"
_UINT64_MAX = (1 << 64) - 1


def _identity(payload: bytes) -> bytes:
    return payload


def _deadline_error(diagnostic: str) -> TransportError:
    return TransportError(
        TransportErrorCode.DEADLINE_EXCEEDED,
        retryable=False,
        diagnostic=diagnostic,
    )


def _rpc_error(error: grpc.aio.AioRpcError) -> TransportError:
    status = error.code()
    if status is grpc.StatusCode.RESOURCE_EXHAUSTED:
        return TransportError(
            TransportErrorCode.BUSY,
            retryable=True,
            diagnostic="the Worker shared-memory endpoint is at capacity",
        )
    if status is grpc.StatusCode.DEADLINE_EXCEEDED:
        return _deadline_error("the Worker shared-memory deadline expired")
    if status is grpc.StatusCode.CANCELLED:
        return TransportError(
            TransportErrorCode.CANCELLED,
            retryable=False,
            diagnostic="the Worker shared-memory call was cancelled",
        )
    if status in {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNKNOWN,
        grpc.StatusCode.ABORTED,
    }:
        return TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            diagnostic=f"the Worker shared-memory call failed with {status.name}",
        )
    if status is grpc.StatusCode.UNIMPLEMENTED:
        return TransportError(
            TransportErrorCode.UNSUPPORTED,
            retryable=False,
            diagnostic="the Worker does not implement shared-memory computation",
        )
    return TransportError(
        TransportErrorCode.PROTOCOL,
        retryable=False,
        diagnostic=f"the Worker shared-memory call failed with {status.name}",
    )


def _selected_hidden_states(batch: WorkerBatch) -> torch.Tensor:
    try:
        if batch.token_indices is None:
            return batch.hidden_states
        return torch.index_select(batch.hidden_states, 0, batch.token_indices)
    except (IndexError, RuntimeError) as error:
        raise TransportProtocolError("Worker batch token indices are invalid") from error


def _copy_request_to_slot(
    batch: WorkerBatch,
    spec: WorkerEndpointConfig,
    buffers: ShmTransferBuffers,
    stream: torch.cuda.Stream | None,
) -> int:
    validate_worker_batch(batch, spec)
    slot = buffers.slot
    if buffers.generation >= _UINT64_MAX:
        raise RuntimeError("shared-memory slot generation is exhausted")
    if buffers.receive_recorded:
        assert buffers.receive_event is not None
        buffers.receive_event.synchronize()
    buffers.generation += 1
    token_count = batch.token_count
    with torch.inference_mode():
        hidden_states = _selected_hidden_states(batch)
        if stream is None:
            slot.hidden_states[:token_count].copy_(hidden_states)
            slot.expert_ids[:token_count].copy_(batch.expert_ids)
            slot.routing_weights[:token_count].copy_(batch.routing_weights)
        else:
            assert buffers.request_copy_event is not None
            with torch.cuda.stream(stream):
                slot.hidden_states[:token_count].copy_(hidden_states, non_blocking=True)
                slot.expert_ids[:token_count].copy_(batch.expert_ids, non_blocking=True)
                slot.routing_weights[:token_count].copy_(
                    batch.routing_weights,
                    non_blocking=True,
                )
                buffers.request_copy_event.record(stream)
                buffers.request_copy_recorded = True
            buffers.request_copy_event.synchronize()
    return buffers.generation


def _copy_slot_to_output(
    token_count: int,
    buffers: ShmTransferBuffers,
    output: torch.Tensor,
    stream: torch.cuda.Stream | None,
) -> None:
    slot = buffers.slot
    source = slot.partial_output[:token_count]
    if stream is None:
        output[:token_count].copy_(source)
        return

    assert buffers.receive_event is not None
    with torch.cuda.stream(stream):
        output[:token_count].copy_(source, non_blocking=True)
        buffers.receive_event.record(stream)
        buffers.receive_recorded = True


def _validate_output(
    output: torch.Tensor,
    batch: WorkerBatch,
    spec: WorkerEndpointConfig,
    device: torch.device,
) -> None:
    if output.shape != (batch.token_count, spec.hidden_dim):
        raise ValueError("shared-memory output must have shape [token_count, hidden_dim]")
    if output.dtype != spec.dtype:
        raise ValueError("shared-memory output dtype does not match the endpoint")
    if output.device != device or output.device != batch.hidden_states.device:
        raise ValueError("shared-memory output and Worker batch must use the configured device")
    if not output.is_contiguous():
        raise ValueError("shared-memory output must be contiguous")


class ShmWorkerTransport(WorkerTransport):
    """Move Tensor bytes through fixed same-host slots and notify with gRPC."""

    def __init__(
        self,
        endpoint: str,
        endpoint_config: WorkerEndpointConfig,
        *,
        max_in_flight: int,
        device: torch.device | str,
        cpu_workers: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if (
            isinstance(max_in_flight, bool)
            or not isinstance(max_in_flight, int)
            or max_in_flight <= 0
        ):
            raise ValueError("max_in_flight must be a positive integer")
        resolved_cpu_workers = cpu_workers if cpu_workers is not None else max_in_flight
        if (
            isinstance(resolved_cpu_workers, bool)
            or not isinstance(resolved_cpu_workers, int)
            or resolved_cpu_workers <= 0
        ):
            raise ValueError("cpu_workers must be a positive integer")

        self._endpoint = endpoint
        self._spec = endpoint_config
        self._device = torch.device(device)
        self._buffers = ShmTransferBufferPool(
            endpoint_config,
            capacity=max_in_flight,
            device=self._device,
        )
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=resolved_cpu_workers,
            thread_name_prefix="expertkit-shm-client",
        )
        self._clock = clock
        self._channel: grpc.aio.Channel | None = None
        self._execute: Any = None
        self._close_session: Any = None
        self._start_lock = asyncio.Lock()
        self._active: set[asyncio.Task[object]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create the control channel and register this process's fixed slots."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("shared-memory Transport is closing")
            if self._channel is not None:
                return
            options = (
                ("grpc.max_send_message_length", SHM_CONTROL_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", SHM_CONTROL_MESSAGE_BYTES),
            )
            channel = grpc.aio.insecure_channel(self._endpoint, options=options)
            open_call = channel.unary_unary(
                _OPEN_METHOD,
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            execute_call = channel.unary_unary(
                _EXECUTE_METHOD,
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            close_call = channel.unary_unary(
                _CLOSE_METHOD,
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            try:
                response = await open_call(
                    encode_open_request(
                        session_id=self._buffers.session_id,
                        segment_name=self._buffers.segment_name,
                        layout=self._buffers.layout,
                        spec=self._spec,
                    ),
                    wait_for_ready=False,
                )
                decode_open_response(response)
            except grpc.aio.AioRpcError as error:
                await channel.close()
                raise _rpc_error(error) from error
            except BaseException:
                await channel.close()
                raise
            self._channel = channel
            self._execute = execute_call
            self._close_session = close_call

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Fill a caller-owned Tensor while moving payloads through shared memory."""

        if self._channel is None or self._execute is None:
            raise RuntimeError("shared-memory Transport has not been started")
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="shared-memory Transport is closing",
            )
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("shared-memory submission requires an asyncio Task")
        self._active.add(task)
        try:
            buffers = await self._acquire(monotonic_deadline)
            try:
                await self._execute_acquired(batch, output, buffers, monotonic_deadline)
            finally:
                self._buffers.put(buffers)
                self._semaphore.release()
        finally:
            self._active.discard(task)

    async def close(self) -> None:
        """Finish submissions, close the Worker mapping, and release resources."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        current = asyncio.current_task()
        active = tuple(task for task in self._active if task is not current)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        close_error: BaseException | None = None
        if self._close_session is not None:
            try:
                response = await self._close_session(
                    encode_close_request(self._buffers.session_id),
                    wait_for_ready=False,
                )
                decode_close_response(response)
            except grpc.aio.AioRpcError as error:
                if error.code() not in {
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.UNAVAILABLE,
                }:
                    close_error = _rpc_error(error)
            except BaseException as error:
                close_error = error
        if self._channel is not None:
            try:
                await self._channel.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._executor.shutdown, wait=True, cancel_futures=True),
            )
        except BaseException as error:
            if close_error is None:
                close_error = error
        try:
            self._buffers.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
        if close_error is not None:
            raise close_error

    async def _acquire(self, monotonic_deadline: float) -> ShmTransferBuffers:
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired before shared-memory admission")
        try:
            if math.isinf(remaining):
                await self._semaphore.acquire()
            else:
                async with asyncio.timeout(remaining):
                    await self._semaphore.acquire()
        except TimeoutError as error:
            raise _deadline_error("deadline expired before shared-memory admission") from error
        try:
            return self._buffers.take()
        except BaseException:
            self._semaphore.release()
            raise

    async def _execute_acquired(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        buffers: ShmTransferBuffers,
        monotonic_deadline: float,
    ) -> None:
        _validate_output(output, batch, self._spec, self._device)
        stream = torch.cuda.current_stream(output.device) if output.device.type == "cuda" else None
        try:
            generation = await self._run_cpu(
                _copy_request_to_slot,
                batch,
                self._spec,
                buffers,
                stream,
            )
        except TransportProtocolError as error:
            raise TransportError(
                TransportErrorCode.INVALID_REQUEST,
                retryable=False,
                diagnostic=str(error),
            ) from error
        slot = buffers.slot
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired before shared-memory notification")
        timeout_micros = (
            _UINT64_MAX
            if math.isinf(remaining)
            else max(1, min(_UINT64_MAX, math.ceil(remaining * 1_000_000)))
        )
        request = encode_execute_request(
            ExecuteSlot(
                session_id=self._buffers.session_id,
                slot_index=slot.index,
                generation=generation,
                layer_id=batch.layer_id,
                topology_version=batch.topology_version,
                token_count=batch.token_count,
                timeout_micros=timeout_micros,
            )
        )
        call = self._execute(
            request,
            timeout=None,
            wait_for_ready=False,
        )
        try:
            if math.isinf(remaining):
                response = await asyncio.shield(call)
            else:
                try:
                    async with asyncio.timeout(remaining):
                        response = await asyncio.shield(call)
                except TimeoutError as error:
                    with suppress(BaseException):
                        await call
                    raise _deadline_error("the Worker shared-memory deadline expired") from error
        except asyncio.CancelledError:
            with suppress(BaseException):
                await call
            raise
        except grpc.aio.AioRpcError as error:
            raise _rpc_error(error) from error

        try:
            decode_execute_response(response, generation, self._spec)
            await self._run_cpu(
                _copy_slot_to_output,
                batch.token_count,
                buffers,
                output,
                stream,
            )
        except TransportProtocolError as error:
            raise TransportError(
                TransportErrorCode.PROTOCOL,
                retryable=False,
                diagnostic=str(error),
            ) from error

    async def _run_cpu(self, function: Callable[..., Any], *args: object) -> Any:
        loop = asyncio.get_running_loop()
        work = loop.run_in_executor(self._executor, function, *args)
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await work
            raise
