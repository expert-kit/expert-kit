"""Frontend NCCL data path with a lightweight gRPC admission stream."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
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
from expertkit_transport.transports.nccl.client_buffers import (
    NcclTransferBufferPool,
    NcclTransferBuffers,
)
from expertkit_transport.transports.nccl.codec import (
    NCCL_CONTROL_MESSAGE_BYTES,
    NcclExecute,
    decode_admitted,
    decode_hello_response,
    decode_terminal,
    encode_execute_request,
    encode_hello_request,
)
from expertkit_transport.transports.nccl.runtime import NcclRuntimeProtocol
from expertkit_transport.transports.validation import validate_worker_batch

_HELLO_METHOD = "/ek.worker.v2.NcclComputationService/Hello"
_EXECUTE_METHOD = "/ek.worker.v2.NcclComputationService/Execute"
_UINT64_MAX = (1 << 64) - 1
_START_TIMEOUT_SECONDS = 10.0


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
            diagnostic="the Worker NCCL control endpoint is at capacity",
        )
    if status is grpc.StatusCode.DEADLINE_EXCEEDED:
        return _deadline_error("the Worker NCCL control deadline expired")
    if status is grpc.StatusCode.CANCELLED:
        return TransportError(
            TransportErrorCode.CANCELLED,
            retryable=False,
            diagnostic="the Worker NCCL control stream was cancelled",
        )
    if status in {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNKNOWN,
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.FAILED_PRECONDITION,
    }:
        return TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=True,
            diagnostic=f"the Worker NCCL control call failed with {status.name}",
        )
    if status is grpc.StatusCode.UNIMPLEMENTED:
        return TransportError(
            TransportErrorCode.UNSUPPORTED,
            retryable=False,
            diagnostic="the Worker does not implement the NCCL control service",
        )
    return TransportError(
        TransportErrorCode.PROTOCOL,
        retryable=False,
        diagnostic=f"the Worker NCCL control call failed with {status.name}",
    )


def _validate_output(
    output: torch.Tensor,
    batch: WorkerBatch,
    spec: WorkerEndpointConfig,
    device: torch.device,
) -> None:
    if output.shape != (batch.token_count, spec.hidden_dim):
        raise ValueError("NCCL output must have shape [token_count, hidden_dim]")
    if output.dtype != spec.dtype:
        raise ValueError("NCCL output dtype does not match the endpoint")
    if output.device != device or batch.hidden_states.device != device:
        raise ValueError("NCCL output and Worker batch must use the configured device")
    if not output.is_contiguous():
        raise ValueError("NCCL output must be contiguous")


class NcclWorkerTransport(WorkerTransport):
    """Move CUDA tensors to one peer in a shared static NCCL world."""

    def __init__(
        self,
        control_endpoint: str,
        endpoint_config: WorkerEndpointConfig,
        *,
        max_in_flight: int,
        device: torch.device | str,
        runtime: NcclRuntimeProtocol,
        peer_rank: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        interceptors: Sequence[grpc.aio.ClientInterceptor] = (),
    ) -> None:
        if not control_endpoint:
            raise ValueError("control_endpoint must not be empty")
        if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int):
            raise ValueError("max_in_flight must be a positive integer")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be a positive integer")
        resolved_device = torch.device(device)
        if resolved_device != runtime.device:
            raise ValueError("NCCL Transport device must match its shared runtime")
        if peer_rank is not None and (
            isinstance(peer_rank, bool)
            or not isinstance(peer_rank, int)
            or not 0 <= peer_rank < runtime.world_size
            or peer_rank == runtime.rank
        ):
            raise ValueError("peer_rank must identify another rank in the NCCL world")

        self._endpoint = control_endpoint
        self._spec = endpoint_config
        self._device = resolved_device
        self._runtime = runtime
        self._peer_rank = peer_rank
        self._clock = clock
        self._interceptors = tuple(interceptors)
        self._buffers = NcclTransferBufferPool(
            endpoint_config,
            device=self._device,
            capacity=max_in_flight,
        )
        self._semaphore = asyncio.Semaphore(max_in_flight)
        # NCCL P2P has no tag.  Only one control admission may exist per peer,
        # even though fixed slots preserve API-level bounded concurrency.
        self._peer_lane = asyncio.Lock()
        self._channel: grpc.aio.Channel | None = None
        self._hello: Any = None
        self._execute: Any = None
        self._sequence = 0
        self._start_lock = asyncio.Lock()
        self._active: set[asyncio.Task[object]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def peer_rank(self) -> int:
        if self._peer_rank is None:
            raise RuntimeError("NCCL Transport has not completed its control handshake")
        return self._peer_rank

    async def start(self) -> None:
        """Start runtime rendezvous in the background and validate the peer."""

        async with self._start_lock:
            if self._closing:
                raise RuntimeError("NCCL Transport is closing")
            if self._channel is not None:
                return
            await self._runtime.start()
            options = (
                ("grpc.max_receive_message_length", NCCL_CONTROL_MESSAGE_BYTES),
                ("grpc.max_send_message_length", NCCL_CONTROL_MESSAGE_BYTES),
            )
            channel = grpc.aio.insecure_channel(
                self._endpoint,
                options=options,
                interceptors=self._interceptors,
            )
            hello = channel.unary_unary(
                _HELLO_METHOD,
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            execute = channel.unary_stream(
                _EXECUTE_METHOD,
                request_serializer=_identity,
                response_deserializer=_identity,
            )
            try:
                response = await hello(
                    encode_hello_request(self._runtime, self._spec),
                    timeout=_START_TIMEOUT_SECONDS,
                    wait_for_ready=False,
                )
                peer = decode_hello_response(response, self._runtime, self._spec)
            except grpc.aio.AioRpcError as error:
                await channel.close()
                raise _rpc_error(error) from error
            except BaseException:
                await channel.close()
                raise
            if self._peer_rank is not None and peer.rank != self._peer_rank:
                await channel.close()
                raise TransportProtocolError(
                    "NCCL control endpoint rank does not match the configured peer"
                )
            self._peer_rank = peer.rank
            self._channel = channel
            self._hello = hello
            self._execute = execute

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        if self._channel is None or self._execute is None or self._peer_rank is None:
            raise RuntimeError("NCCL Transport has not been started")
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="NCCL Transport is closing",
            )
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("NCCL submission requires an asyncio Task")
        self._active.add(task)
        try:
            await self._acquire(self._semaphore, monotonic_deadline, "NCCL admission")
            buffers: NcclTransferBuffers | None = None
            try:
                buffers = self._buffers.take()
                await self._acquire(
                    self._peer_lane,
                    monotonic_deadline,
                    "NCCL peer ordering",
                )
                try:
                    if self._closing:
                        raise TransportError(
                            TransportErrorCode.UNAVAILABLE,
                            retryable=True,
                            diagnostic="NCCL Transport is closing",
                        )
                    await self._execute_acquired(
                        batch,
                        output,
                        buffers,
                        monotonic_deadline,
                    )
                finally:
                    self._peer_lane.release()
            finally:
                if buffers is not None:
                    self._buffers.put(buffers)
                self._semaphore.release()
        finally:
            self._active.discard(task)

    async def close(self) -> None:
        if self._close_task is None:
            # Publish the admission gate before scheduling cleanup.  Otherwise
            # a caller awakened in the same loop tick can enter behind an
            # in-flight untagged exchange and deadlock shutdown.
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        self._closing = True
        current = asyncio.current_task()
        active = tuple(task for task in self._active if task is not current)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        if self._channel is not None:
            await self._channel.close()
        self._channel = None
        self._hello = None
        self._execute = None
        self._buffers.close()

    async def _execute_acquired(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        buffers: NcclTransferBuffers,
        monotonic_deadline: float,
    ) -> None:
        _validate_output(output, batch, self._spec, self._device)
        try:
            validate_worker_batch(batch, self._spec)
            await self._runtime.wait_ready(monotonic_deadline=monotonic_deadline)
            await self._pack(batch, buffers)
        except TransportProtocolError as error:
            raise TransportError(
                TransportErrorCode.INVALID_REQUEST,
                retryable=False,
                diagnostic=str(error),
            ) from error
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired before NCCL control admission")
        self._sequence += 1
        sequence = self._sequence
        timeout_micros = (
            _UINT64_MAX
            if math.isinf(remaining)
            else max(1, min(_UINT64_MAX, math.ceil(remaining * 1_000_000)))
        )
        request = encode_execute_request(
            NcclExecute(
                sequence=sequence,
                client_rank=self._runtime.rank,
                layer_id=batch.layer_id,
                topology_version=batch.topology_version,
                token_count=batch.token_count,
                timeout_micros=timeout_micros,
                distinct_expert_ids=batch.distinct_expert_ids,
            ),
            self._spec,
        )
        call = self._execute(
            request,
            # The request metadata carries the caller SLA.  A gRPC deadline can
            # expire after the Worker has committed its admission ACK but before
            # this process observes it, leaving an already-posted NCCL receive
            # without its matching sends.  Keep the control stream alive through
            # the mandatory matching phase and report the metadata deadline once
            # the untagged operation sequence is closed.
            timeout=None,
            wait_for_ready=False,
        )
        admitted = False
        caller_cancelled = False
        exchange_deadline_error: TransportError | None = None
        try:
            first_read = asyncio.create_task(call.read())
            try:
                first = await asyncio.shield(first_read)
            except asyncio.CancelledError:
                caller_cancelled = True
                first = await first_read
            if first is grpc.aio.EOF:
                raise TransportProtocolError("NCCL control stream ended before admission")
            try:
                decode_admitted(first, sequence)
            except TransportProtocolError as admission_error:
                try:
                    decode_terminal(first, sequence, self._spec)
                except TransportError:
                    raise
                raise admission_error
            admitted = True

            token_count = batch.token_count
            try:
                await self._runtime.exchange(
                    self.peer_rank,
                    buffers.hidden_states[:token_count],
                    buffers.expert_ids[:token_count],
                    buffers.routing_weights[:token_count],
                    buffers.partial_output[:token_count],
                    # Admission ACK makes all four untagged operations mandatory;
                    # report the SLA only after the matching phase is closed.
                    monotonic_deadline=math.inf,
                )
            except asyncio.CancelledError:
                caller_cancelled = True
            except TransportError as error:
                if error.code != TransportErrorCode.DEADLINE_EXCEEDED:
                    raise
                exchange_deadline_error = error
            if self._clock() >= monotonic_deadline and exchange_deadline_error is None:
                exchange_deadline_error = _deadline_error(
                    "deadline expired during the mandatory NCCL exchange"
                )

            terminal_read = asyncio.create_task(call.read())
            try:
                terminal = await asyncio.shield(terminal_read)
            except asyncio.CancelledError:
                caller_cancelled = True
                terminal = await terminal_read
            if terminal is grpc.aio.EOF:
                raise TransportProtocolError("NCCL control stream ended before completion")
            decode_terminal(terminal, sequence, self._spec)
            trailing_read = asyncio.create_task(call.read())
            try:
                trailing = await asyncio.shield(trailing_read)
            except asyncio.CancelledError:
                caller_cancelled = True
                trailing = await trailing_read
            if trailing is not grpc.aio.EOF:
                raise TransportProtocolError("NCCL control stream contains extra messages")
            if exchange_deadline_error is not None:
                raise exchange_deadline_error
            output.copy_(buffers.partial_output[:token_count])
            if self._device.type == "cuda":
                assert buffers.copy_event is not None
                buffers.copy_event.record(torch.cuda.current_stream(self._device))
                buffers.copy_recorded = True
                copy_wait = asyncio.create_task(asyncio.to_thread(buffers.copy_event.synchronize))
                try:
                    await asyncio.shield(copy_wait)
                except asyncio.CancelledError:
                    caller_cancelled = True
                    await copy_wait
            if caller_cancelled:
                raise asyncio.CancelledError
        except grpc.aio.AioRpcError as error:
            raise _rpc_error(error) from error
        except TransportProtocolError as error:
            raise TransportError(
                TransportErrorCode.PROTOCOL,
                retryable=False,
                diagnostic=str(error),
            ) from error
        except asyncio.CancelledError:
            if not admitted:
                call.cancel()
                with suppress(BaseException):
                    await call
            raise
        except BaseException:
            call.cancel()
            raise

    async def _pack(self, batch: WorkerBatch, buffers: NcclTransferBuffers) -> None:
        token_count = batch.token_count
        with torch.inference_mode():
            if batch.token_indices is None:
                buffers.hidden_states[:token_count].copy_(batch.hidden_states)
            else:
                try:
                    torch.index_select(
                        batch.hidden_states,
                        0,
                        batch.token_indices,
                        out=buffers.hidden_states[:token_count],
                    )
                except (IndexError, RuntimeError) as error:
                    raise TransportProtocolError(
                        "Worker batch token indices are invalid"
                    ) from error
            buffers.expert_ids[:token_count].copy_(batch.expert_ids)
            buffers.routing_weights[:token_count].copy_(batch.routing_weights)
            if self._device.type == "cuda":
                assert buffers.copy_event is not None
                buffers.copy_event.record(torch.cuda.current_stream(self._device))
                buffers.copy_recorded = True
                copy_wait = asyncio.create_task(asyncio.to_thread(buffers.copy_event.synchronize))
                try:
                    await asyncio.shield(copy_wait)
                except asyncio.CancelledError:
                    await copy_wait
                    raise

    async def _acquire(
        self,
        semaphore: asyncio.Semaphore | asyncio.Lock,
        monotonic_deadline: float,
        subject: str,
    ) -> None:
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error(f"deadline expired before {subject}")
        try:
            if math.isinf(remaining):
                await semaphore.acquire()
            else:
                async with asyncio.timeout(remaining):
                    await semaphore.acquire()
        except TimeoutError as error:
            raise _deadline_error(f"deadline expired before {subject}") from error


__all__ = ["NcclWorkerTransport"]
