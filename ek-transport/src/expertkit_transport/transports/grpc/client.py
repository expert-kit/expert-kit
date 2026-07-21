"""Frontend-side asynchronous unary gRPC Worker Transport."""

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
from expertkit_transport.transports.grpc.buffers import (
    GrpcTransferBufferPool,
    GrpcTransferBuffers,
)
from expertkit_transport.transports.grpc.codec import (
    _serialize_host_request,
    decode_response,
)
from expertkit_transport.transports.grpc.spec import calculate_message_limits
from expertkit_transport.transports.validation import validate_worker_batch

_EXECUTE_METHOD = "/ek.worker.v2.ComputationService/Execute"


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
            diagnostic="the Worker gRPC endpoint is at capacity",
        )
    if status is grpc.StatusCode.DEADLINE_EXCEEDED:
        return _deadline_error("the Worker gRPC deadline expired")
    if status is grpc.StatusCode.CANCELLED:
        return TransportError(
            TransportErrorCode.CANCELLED,
            retryable=False,
            diagnostic="the Worker gRPC call was cancelled",
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
            diagnostic=f"the Worker gRPC call failed with {status.name}",
        )
    if status is grpc.StatusCode.UNIMPLEMENTED:
        return TransportError(
            TransportErrorCode.UNSUPPORTED,
            retryable=False,
            diagnostic="the Worker does not implement the v2 computation method",
        )
    return TransportError(
        TransportErrorCode.PROTOCOL,
        retryable=False,
        diagnostic=f"the Worker gRPC call failed with {status.name}",
    )


def _selected_hidden_states(batch: WorkerBatch) -> torch.Tensor:
    try:
        if batch.token_indices is None:
            return batch.hidden_states
        return torch.index_select(batch.hidden_states, 0, batch.token_indices)
    except (IndexError, RuntimeError) as error:
        raise TransportProtocolError("Worker batch token indices are invalid") from error


def _encode_with_staging(
    batch: WorkerBatch,
    spec: WorkerEndpointConfig,
    buffers: GrpcTransferBuffers,
    stream: torch.cuda.Stream | None,
) -> bytes:
    validate_worker_batch(batch, spec)
    token_count = batch.token_count
    with torch.inference_mode():
        if stream is None:
            hidden_states = _selected_hidden_states(batch)
            buffers.host_hidden_states[:token_count].copy_(hidden_states)
            buffers.host_expert_ids[:token_count].copy_(batch.expert_ids)
            buffers.host_routing_weights[:token_count].copy_(batch.routing_weights)
        else:
            assert buffers.request_copy_event is not None
            with torch.cuda.stream(stream):
                hidden_states = _selected_hidden_states(batch)
                buffers.host_hidden_states[:token_count].copy_(
                    hidden_states,
                    non_blocking=True,
                )
                buffers.host_expert_ids[:token_count].copy_(
                    batch.expert_ids,
                    non_blocking=True,
                )
                buffers.host_routing_weights[:token_count].copy_(
                    batch.routing_weights,
                    non_blocking=True,
                )
                buffers.request_copy_event.record(stream)
                buffers.request_copy_recorded = True
            buffers.request_copy_event.synchronize()

    return _serialize_host_request(
        batch,
        spec,
        buffers.host_hidden_states[:token_count],
        buffers.host_expert_ids[:token_count],
        buffers.host_routing_weights[:token_count],
    )


def _decode_into_output(
    payload: bytes,
    token_count: int,
    spec: WorkerEndpointConfig,
    buffers: GrpcTransferBuffers,
    output: torch.Tensor,
    stream: torch.cuda.Stream | None,
) -> None:
    partial_output = decode_response(payload, token_count, spec)
    if stream is None:
        output[:token_count].copy_(partial_output)
        return

    assert buffers.receive_event is not None
    assert buffers.host_partial_output is not None
    if buffers.receive_recorded:
        buffers.receive_event.synchronize()
    buffers.host_partial_output[:token_count].copy_(partial_output)
    with torch.cuda.stream(stream):
        output[:token_count].copy_(
            buffers.host_partial_output[:token_count],
            non_blocking=True,
        )
        buffers.receive_event.record(stream)
        buffers.receive_recorded = True


def _validate_output(
    output: torch.Tensor,
    batch: WorkerBatch,
    spec: WorkerEndpointConfig,
    device: torch.device,
) -> None:
    if output.shape != (batch.token_count, spec.hidden_dim):
        raise ValueError("gRPC output must have shape [token_count, hidden_dim]")
    if output.dtype != spec.dtype:
        raise ValueError("gRPC output dtype does not match the endpoint")
    if output.device != device or output.device != batch.hidden_states.device:
        raise ValueError("gRPC output and Worker batch must use the configured device")
    if not output.is_contiguous():
        raise ValueError("gRPC output must be contiguous")


class GrpcWorkerTransport(WorkerTransport):
    """Send bounded asynchronous unary calls to one plaintext Worker endpoint."""

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
        if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int):
            raise ValueError("max_in_flight must be a positive integer")
        if max_in_flight <= 0:
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
        self._limits = calculate_message_limits(endpoint_config)
        self._buffers = GrpcTransferBufferPool(
            endpoint_config,
            device=self._device,
            capacity=max_in_flight,
        )
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=resolved_cpu_workers,
            thread_name_prefix="expertkit-grpc-client",
        )
        self._clock = clock
        self._channel: grpc.aio.Channel | None = None
        self._execute: Any = None
        self._start_lock = asyncio.Lock()
        self._active: set[asyncio.Task[object]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create the reusable plaintext `grpc.aio` channel."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("gRPC Transport is closing")
            if self._channel is not None:
                return
            channel = grpc.aio.insecure_channel(
                self._endpoint,
                options=self._limits.client_options,
            )
            self._channel = channel
            self._execute = channel.unary_unary(
                _EXECUTE_METHOD,
                request_serializer=_identity,
                response_deserializer=_identity,
            )

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Fill a caller-owned Tensor without blocking the asyncio event loop."""

        if self._channel is None or self._execute is None:
            raise RuntimeError("gRPC Transport has not been started")
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="gRPC Transport is closing",
            )
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("gRPC submission requires an asyncio Task")
        self._active.add(task)
        try:
            buffers = await self._acquire(monotonic_deadline)
            try:
                if self._closing:
                    raise TransportError(
                        TransportErrorCode.UNAVAILABLE,
                        retryable=True,
                        diagnostic="gRPC Transport is closing",
                    )
                await self._execute_acquired(batch, output, buffers, monotonic_deadline)
            finally:
                self._buffers.put(buffers)
                self._semaphore.release()
        finally:
            self._active.discard(task)

    async def close(self) -> None:
        """Cancel active RPCs and release the channel and bounded CPU workers."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        if self._channel is not None:
            await self._channel.close()
        current = asyncio.current_task()
        active = tuple(task for task in self._active if task is not current)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._executor.shutdown, wait=True, cancel_futures=True),
        )
        self._buffers.close()

    async def _acquire(self, monotonic_deadline: float) -> GrpcTransferBuffers:
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired before gRPC admission")
        try:
            if math.isinf(remaining):
                await self._semaphore.acquire()
            else:
                async with asyncio.timeout(remaining):
                    await self._semaphore.acquire()
        except TimeoutError as error:
            raise _deadline_error("deadline expired before gRPC admission") from error
        try:
            return self._buffers.take()
        except BaseException:
            self._semaphore.release()
            raise

    async def _execute_acquired(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        buffers: GrpcTransferBuffers,
        monotonic_deadline: float,
    ) -> None:
        _validate_output(output, batch, self._spec, self._device)
        stream = torch.cuda.current_stream(output.device) if output.device.type == "cuda" else None
        try:
            request = await self._run_cpu(
                _encode_with_staging,
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

        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired before the Worker gRPC call")
        call = self._execute(
            request,
            timeout=None if math.isinf(remaining) else remaining,
            wait_for_ready=False,
        )
        try:
            response = await call
        except asyncio.CancelledError:
            call.cancel()
            with suppress(BaseException):
                await call
            raise
        except grpc.aio.AioRpcError as error:
            raise _rpc_error(error) from error

        try:
            await self._run_cpu(
                _decode_into_output,
                response,
                batch.token_count,
                self._spec,
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
