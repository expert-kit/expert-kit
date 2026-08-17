"""Frontend Transfer Engine data path with gRPC admission and completion."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable, Sequence
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
from expertkit_transport.transports.transfer_engine.arena import (
    TransferArena,
    TransferArenaDescriptor,
    TransferArenaSlot,
)
from expertkit_transport.transports.transfer_engine.codec import (
    TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES,
    TransferExecutePull,
    TransferOpenSession,
    decode_admitted,
    decode_open_session_response,
    decode_terminal,
    encode_close_session,
    encode_execute_pull_request,
    encode_open_session_request,
)
from expertkit_transport.transports.transfer_engine.runtime import (
    TransferEngineRuntimeProtocol,
)
from expertkit_transport.transports.validation import validate_worker_batch

_OPEN_METHOD = "/ek.worker.v2.TransferEngineComputationService/OpenSession"
_EXECUTE_METHOD = "/ek.worker.v2.TransferEngineComputationService/ExecutePull"
_CLOSE_METHOD = "/ek.worker.v2.TransferEngineComputationService/CloseSession"
_UINT64_MAX = (1 << 64) - 1
_START_TIMEOUT_SECONDS = 10.0
_TERMINAL_DRAIN_GRACE_SECONDS = 30.0
_UNSAFE_STAGING_GRAPHS: list[tuple[object, ...]] = []
_UNSAFE_CLOSE_GRAPHS: list[tuple[object, ...]] = []


def _copy_tensor(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Issue one staging copy through a narrow fault-injection seam."""

    destination.copy_(source)


class _UnsafeCudaStaging(RuntimeError):
    """Mark caller-owned CUDA memory whose last access cannot be fenced."""

    def __init__(self, phase: str, cause: BaseException) -> None:
        super().__init__(phase)
        self.phase = phase
        self.__cause__ = cause


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
            diagnostic="the Worker Transfer Engine endpoint is at capacity",
        )
    if status is grpc.StatusCode.DEADLINE_EXCEEDED:
        return _deadline_error("the Worker Transfer Engine control deadline expired")
    if status is grpc.StatusCode.CANCELLED:
        return TransportError(
            TransportErrorCode.CANCELLED,
            retryable=False,
            diagnostic="the Worker Transfer Engine control stream was cancelled",
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
            diagnostic=f"the Worker Transfer Engine call failed with {status.name}",
        )
    if status is grpc.StatusCode.UNIMPLEMENTED:
        return TransportError(
            TransportErrorCode.UNSUPPORTED,
            retryable=False,
            diagnostic="the Worker does not implement Transfer Engine transport",
        )
    return TransportError(
        TransportErrorCode.PROTOCOL,
        retryable=False,
        diagnostic=f"the Worker Transfer Engine call failed with {status.name}",
    )


class _AmbiguousRemoteState(RuntimeError):
    def __init__(self, error: TransportError) -> None:
        super().__init__(error.diagnostic)
        self.error = error


class TransferEngineWorkerTransport(WorkerTransport):
    """Expose a fixed local arena and let one Worker pull each admitted request."""

    def __init__(
        self,
        control_endpoint: str,
        endpoint_config: WorkerEndpointConfig,
        *,
        max_in_flight: int,
        device: torch.device | str,
        runtime: TransferEngineRuntimeProtocol,
        expected_worker_start_id: str,
        clock: Callable[[], float] = time.monotonic,
        interceptors: Sequence[grpc.aio.ClientInterceptor] = (),
    ) -> None:
        if not control_endpoint:
            raise ValueError("control_endpoint must not be empty")
        if (
            isinstance(max_in_flight, bool)
            or not isinstance(max_in_flight, int)
            or max_in_flight <= 0
        ):
            raise ValueError("max_in_flight must be a positive integer")
        if not isinstance(expected_worker_start_id, str) or not expected_worker_start_id:
            raise ValueError("expected_worker_start_id must not be empty")
        resolved_device = torch.device(device)
        if resolved_device != runtime.device:
            raise ValueError("Transfer Engine device must match its shared runtime")
        self._endpoint = control_endpoint
        self._spec = endpoint_config
        self._device = resolved_device
        self._runtime = runtime
        self._worker_start_id = expected_worker_start_id
        self._clock = clock
        self._interceptors = tuple(interceptors)
        self._arena = TransferArena(
            endpoint_config,
            device=self._device,
            capacity=max_in_flight,
        )
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._client_epoch = uuid.uuid4().hex
        self._worker_arena: TransferArenaDescriptor | None = None
        self._worker_session_nonce: str | None = None
        self._channel: grpc.aio.Channel | None = None
        self._open: Any = None
        self._execute: Any = None
        self._close_session: Any = None
        self._sequence = 0
        self._registered = False
        self._start_lock = asyncio.Lock()
        self._active: set[asyncio.Task[object]] = set()
        self._quarantined_slots: set[TransferArenaSlot] = set()
        self._retain_arena = False
        self._poisoned_diagnostic: str | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def worker_arena(self) -> TransferArenaDescriptor:
        if self._worker_arena is None:
            raise RuntimeError("Transfer Engine Transport has not completed OpenSession")
        return self._worker_arena

    async def start(self) -> None:
        async with self._start_lock:
            if self._closing:
                raise RuntimeError("Transfer Engine Transport is closing")
            if self._channel is not None:
                return
            await self._runtime.start()
            request = TransferOpenSession(
                client_epoch=self._client_epoch,
                client_session_id=self._runtime.session_id,
                client_runtime_generation=self._runtime.generation,
                expected_worker_start_id=self._worker_start_id,
                backend=self._runtime.backend,
                arena=self._arena.descriptor,
            )
            encoded_request = encode_open_session_request(request, self._spec)
            await self._runtime.register_tensor(self._arena.slab)
            self._registered = True
            options = (
                ("grpc.max_receive_message_length", TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES),
                ("grpc.max_send_message_length", TRANSFER_ENGINE_CONTROL_MESSAGE_BYTES),
            )
            try:
                channel = grpc.aio.insecure_channel(
                    self._endpoint,
                    options=options,
                    interceptors=self._interceptors,
                )
                open_session = channel.unary_unary(
                    _OPEN_METHOD,
                    request_serializer=_identity,
                    response_deserializer=_identity,
                )
                execute = channel.unary_stream(
                    _EXECUTE_METHOD,
                    request_serializer=_identity,
                    response_deserializer=_identity,
                )
                close_session = channel.unary_unary(
                    _CLOSE_METHOD,
                    request_serializer=_identity,
                    response_deserializer=_identity,
                )
            except BaseException:
                await self._rollback_registration()
                raise
            try:
                response = await open_session(
                    encoded_request,
                    timeout=_START_TIMEOUT_SECONDS,
                    wait_for_ready=False,
                )
                result = decode_open_session_response(
                    response,
                    self._spec,
                    self._worker_start_id,
                    self._runtime.backend,
                )
            except grpc.aio.AioRpcError as error:
                mapped = _rpc_error(error)
                if error.code() in {
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.UNIMPLEMENTED,
                }:
                    await channel.close()
                    await self._rollback_registration()
                    raise mapped from error
                diagnostic = (
                    "Transfer Engine OpenSession may have committed without a visible "
                    "acknowledgement; the registered arena is retained and this process "
                    "must restart"
                )
                self._retain_arena = True
                self._poisoned_diagnostic = diagnostic
                self._runtime.quarantine(diagnostic)
                await channel.close()
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=False,
                    diagnostic=f"{diagnostic}: {mapped.diagnostic}",
                ) from error
            except BaseException as error:
                diagnostic = (
                    "Transfer Engine OpenSession response could not be proven; the "
                    "registered arena is retained and this process must restart"
                )
                self._retain_arena = True
                self._poisoned_diagnostic = diagnostic
                self._runtime.quarantine(diagnostic)
                await channel.close()
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=False,
                    diagnostic=diagnostic,
                ) from error
            self._worker_arena = result.arena
            self._worker_session_nonce = result.session_nonce
            self._channel = channel
            self._open = open_session
            self._execute = execute
            self._close_session = close_session

    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        if self._channel is None or self._execute is None:
            raise RuntimeError("Transfer Engine Transport has not been started")
        if self._closing:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=True,
                diagnostic="Transfer Engine Transport is closing",
            )
        if self._poisoned_diagnostic is not None:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic=self._poisoned_diagnostic,
            )
        self._runtime.ensure_healthy()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Transfer Engine submission requires an asyncio Task")
        self._active.add(task)
        try:
            await self._acquire(monotonic_deadline)
            slot: TransferArenaSlot | None = None
            reusable = True
            try:
                slot = self._arena.take()
                if slot is None:
                    raise RuntimeError("Transfer Engine admission and arena diverged")
                await self._execute_slot(batch, output, slot, monotonic_deadline)
            except _AmbiguousRemoteState as ambiguous:
                reusable = False
                if slot is not None:
                    self._quarantined_slots.add(slot)
                diagnostic = (
                    "Transfer Engine lost a provable terminal state after request dispatch; "
                    "the registered arena is quarantined and this process must restart"
                )
                self._poisoned_diagnostic = diagnostic
                self._runtime.quarantine(diagnostic)
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=False,
                    diagnostic=f"{diagnostic}: {ambiguous.error.diagnostic}",
                ) from ambiguous
            except BaseException:
                try:
                    self._runtime.ensure_healthy()
                except TransportError:
                    reusable = False
                    self._retain_arena = True
                    if slot is not None:
                        self._quarantined_slots.add(slot)
                raise
            finally:
                if slot is not None and reusable:
                    self._arena.put(slot)
                self._semaphore.release()
        finally:
            self._active.discard(task)

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _execute_slot(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        slot: TransferArenaSlot,
        monotonic_deadline: float,
    ) -> None:
        self._validate_output(output, batch)
        try:
            validate_worker_batch(batch, self._spec)
            await self._pack(batch, slot, monotonic_deadline)
        except _UnsafeCudaStaging as unsafe:
            raise self._quarantine_cuda_staging(
                unsafe.phase,
                batch,
                output,
                slot,
                unsafe_output=False,
            ) from unsafe.__cause__
        except TransportProtocolError as error:
            raise TransportError(
                TransportErrorCode.INVALID_REQUEST,
                retryable=False,
                diagnostic=str(error),
            ) from error
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired before Transfer Engine admission")
        if not math.isfinite(remaining):
            raise ValueError("Transfer Engine execution requires a finite monotonic deadline")
        self._sequence += 1
        sequence = self._sequence
        timeout_micros = (
            _UINT64_MAX
            if math.isinf(remaining)
            else max(1, min(_UINT64_MAX, math.ceil(remaining * 1_000_000)))
        )
        request = TransferExecutePull(
            sequence=sequence,
            client_epoch=self._client_epoch,
            session_nonce=self._require_worker_session_nonce(),
            expected_worker_start_id=self._worker_start_id,
            client_slot_index=slot.index,
            client_slot_generation=slot.generation,
            layer_id=batch.layer_id,
            topology_version=batch.topology_version,
            token_count=batch.token_count,
            timeout_micros=timeout_micros,
            distinct_expert_ids=batch.distinct_expert_ids,
        )
        self._runtime.ensure_healthy()
        call = self._execute(
            encode_execute_pull_request(request, self._spec),
            # Keep the control stream alive briefly after the request deadline so
            # the Worker can drain an already-started DMA and report its terminal
            # state.  The finite RPC deadline also bounds a missing first frame.
            timeout=remaining + _TERMINAL_DRAIN_GRACE_SECONDS,
            wait_for_ready=False,
        )
        terminal_confirmed = False
        try:
            first = await call.read()
            if first is grpc.aio.EOF:
                raise TransportProtocolError("Transfer Engine stream ended before admission")
            try:
                decode_admitted(first, sequence)
            except TransportProtocolError as admission_error:
                try:
                    decode_terminal(first, sequence, self._spec)
                except TransportError:
                    # A first-frame terminal error is an ordered, explicit
                    # pre-admission rejection.  No remote DMA can follow it.
                    terminal_confirmed = True
                    raise
                raise admission_error

            terminal = await call.read()
            if terminal is grpc.aio.EOF:
                raise TransportProtocolError("Transfer Engine stream ended before completion")
            try:
                decode_terminal(terminal, sequence, self._spec)
            except TransportError:
                await self._runtime.acquire_remote_writes(
                    monotonic_deadline=math.inf,
                )
                terminal_confirmed = True
                raise
            await self._runtime.acquire_remote_writes(
                monotonic_deadline=math.inf,
            )
            terminal_confirmed = True
            trailing = await call.read()
            if trailing is not grpc.aio.EOF:
                raise TransportProtocolError("Transfer Engine stream contains extra messages")

            try:
                await self._unpack_output(batch, output, slot)
            except _UnsafeCudaStaging as unsafe:
                raise self._quarantine_cuda_staging(
                    unsafe.phase,
                    batch,
                    output,
                    slot,
                    unsafe_output=True,
                ) from unsafe.__cause__
            if self._clock() >= monotonic_deadline:
                raise _deadline_error("deadline expired during Transfer Engine execution")
        except grpc.aio.AioRpcError as error:
            mapped = _rpc_error(error)
            if not terminal_confirmed:
                raise _AmbiguousRemoteState(mapped) from error
            raise mapped from error
        except TransportProtocolError as error:
            if not terminal_confirmed:
                call.cancel()
                raise _AmbiguousRemoteState(
                    TransportError(
                        TransportErrorCode.UNAVAILABLE,
                        retryable=False,
                        diagnostic=str(error),
                    )
                ) from error
            raise TransportError(
                TransportErrorCode.PROTOCOL,
                retryable=False,
                diagnostic=str(error),
            ) from error
        except TransportError:
            raise
        except asyncio.CancelledError:
            if not terminal_confirmed:
                call.cancel()
                raise _AmbiguousRemoteState(
                    TransportError(
                        TransportErrorCode.CANCELLED,
                        retryable=False,
                        diagnostic=(
                            "the caller cancelled before the Worker terminal state was observed"
                        ),
                    )
                ) from None
            raise
        except BaseException as error:
            call.cancel()
            if not terminal_confirmed:
                raise _AmbiguousRemoteState(
                    TransportError(
                        TransportErrorCode.UNAVAILABLE,
                        retryable=False,
                        diagnostic=(
                            "the Transfer Engine control stream failed before a terminal state"
                        ),
                    )
                ) from error
            raise

    async def _pack(
        self,
        batch: WorkerBatch,
        slot: TransferArenaSlot,
        monotonic_deadline: float,
    ) -> None:
        token_count = batch.token_count
        with torch.inference_mode():
            try:
                if batch.token_indices is None:
                    _copy_tensor(slot.hidden_states[:token_count], batch.hidden_states)
                else:
                    torch.index_select(
                        batch.hidden_states,
                        0,
                        batch.token_indices,
                        out=slot.hidden_states[:token_count],
                    )
                _copy_tensor(slot.expert_ids[:token_count], batch.expert_ids)
                _copy_tensor(slot.routing_weights[:token_count], batch.routing_weights)
                if slot.copy_event is not None:
                    slot.copy_event.record(torch.cuda.current_stream(self._device))
            except BaseException as error:
                if slot.copy_event is not None:
                    raise _UnsafeCudaStaging("input-copy", error) from error
                if isinstance(error, IndexError | RuntimeError):
                    raise TransportProtocolError(
                        "Worker batch token indices or CUDA staging copy is invalid"
                    ) from error
                raise
            if slot.copy_event is not None:
                try:
                    await self._runtime.wait_event(
                        slot.copy_event,
                        monotonic_deadline=monotonic_deadline,
                    )
                except asyncio.CancelledError:
                    # wait_event defers cancellation until the recorded event is
                    # terminal, so caller-owned inputs are safe to release.
                    raise
                except TransportError as error:
                    try:
                        self._runtime.ensure_healthy()
                    except TransportError:
                        raise _UnsafeCudaStaging("input-copy fence", error) from error
                    raise
                except BaseException as error:
                    if slot.copy_event is not None:
                        raise _UnsafeCudaStaging("input-copy fence", error) from error
                    raise

    async def _unpack_output(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        slot: TransferArenaSlot,
    ) -> None:
        try:
            _copy_tensor(output, slot.partial_output[: batch.token_count])
            if slot.copy_event is not None:
                slot.copy_event.record(torch.cuda.current_stream(self._device))
        except BaseException as error:
            if slot.copy_event is not None:
                raise _UnsafeCudaStaging("output-copy", error) from error
            raise
        if slot.copy_event is None:
            return
        try:
            await self._runtime.wait_event(
                slot.copy_event,
                monotonic_deadline=math.inf,
            )
        except asyncio.CancelledError:
            # wait_event completes the event barrier before surfacing cancellation.
            raise
        except TransportError as error:
            try:
                self._runtime.ensure_healthy()
            except TransportError:
                raise _UnsafeCudaStaging("output-copy fence", error) from error
            raise
        except BaseException as error:
            if slot.copy_event is not None:
                raise _UnsafeCudaStaging("output-copy fence", error) from error
            raise

    def _quarantine_cuda_staging(
        self,
        phase: str,
        batch: WorkerBatch,
        output: torch.Tensor,
        slot: TransferArenaSlot,
        *,
        unsafe_output: bool,
    ) -> TransportError:
        diagnostic = (
            f"CUDA {phase} completion could not be proven; caller-owned Tensor "
            "storage is retained and this process must restart"
        )
        self._retain_arena = True
        self._poisoned_diagnostic = diagnostic
        self._runtime.quarantine(diagnostic)
        # A quarantined runtime retains registered slabs, but it does not own
        # caller input/output storage. Keep the entire Python ownership graph
        # alive process-wide because no later CUDA API can safely fence it.
        _UNSAFE_STAGING_GRAPHS.append((self, self._runtime, self._arena, slot, batch, output))
        return TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=False,
            unsafe_tensor_ownership=True,
            unsafe_output=unsafe_output,
            diagnostic=diagnostic,
        )

    async def _acquire(self, monotonic_deadline: float) -> None:
        remaining = monotonic_deadline - self._clock()
        if remaining <= 0:
            raise _deadline_error("deadline expired while waiting for Transfer Engine admission")
        try:
            if math.isinf(remaining):
                await self._semaphore.acquire()
            else:
                async with asyncio.timeout(remaining):
                    await self._semaphore.acquire()
        except TimeoutError as error:
            raise _deadline_error(
                "deadline expired while waiting for Transfer Engine admission"
            ) from error

    async def _close(self) -> None:
        current = asyncio.current_task()
        active = tuple(task for task in self._active if task is not current)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        close_failure: BaseException | None = None
        can_release_arena = not self._quarantined_slots and not self._retain_arena
        if (
            can_release_arena
            and self._close_session is not None
            and self._worker_session_nonce is not None
        ):
            try:
                # Prepare gates the complete Mooncake target and drains every
                # sibling before ACK. Keep that gate held while this arena is
                # deregistered so no surviving sibling can reach stale backend
                # state in the deregistration/commit window.
                await self._close_phase("prepare")
                await self._rollback_registration()
                # Commit performs backend-specific retirement, retires this
                # epoch, and only then lets sibling sessions execute again.
                # Intra-NVLink and RDMA invalidate their backend-specific
                # remote-target caches after synchronous deregistration.
                await self._close_phase("commit")
            except BaseException as error:
                close_failure = error
                diagnostic = (
                    "Transfer Engine two-phase CloseSession could not prove target gating, "
                    "arena deregistration, and backend retirement; the arena is retained "
                    "and this process must restart"
                )
                self._retain_arena = True
                self._poisoned_diagnostic = diagnostic
                # Deregistration may already have committed before CommitClose
                # or its ACK failed. The runtime then no longer owns this slab,
                # while the Worker may still retain backend-specific peer state.
                # Retain the complete ownership graph independently of topology
                # references until process exit.
                _UNSAFE_CLOSE_GRAPHS.append(
                    (
                        self,
                        self._runtime,
                        self._arena,
                        self._arena.slab,
                        self._close_task,
                        self._channel,
                    )
                )
                self._runtime.quarantine(diagnostic)
        channel_failure: BaseException | None = None
        if self._channel is not None:
            try:
                await self._channel.close()
            except BaseException as error:
                channel_failure = error
        self._channel = None
        self._open = None
        self._execute = None
        self._close_session = None
        self._worker_session_nonce = None
        if self._quarantined_slots or self._retain_arena:
            if close_failure is not None:
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=False,
                    diagnostic=self._poisoned_diagnostic or str(close_failure),
                ) from close_failure
            return
        self._arena.close()
        await self._rollback_registration()
        if channel_failure is not None:
            raise channel_failure

    async def _close_phase(self, phase: str) -> None:
        if self._close_session is None or self._worker_session_nonce is None:
            raise RuntimeError("Transfer Engine CloseSession channel is unavailable")
        response = await self._close_session(
            encode_close_session(
                phase,
                self._client_epoch,
                self._runtime.session_id,
                self._runtime.generation,
                self._worker_start_id,
                self._worker_session_nonce,
            ),
            # This is a memory-lifetime barrier. Worker shutdown owns the
            # bounded listener grace; a client-side timeout would turn a safe
            # in-progress drain/invalidation into an ambiguous release.
            timeout=None,
            wait_for_ready=False,
        )
        if response != b"":
            raise TransportProtocolError(
                f"Transfer Engine {phase} CloseSession returned an invalid acknowledgement"
            )

    async def _rollback_registration(self) -> None:
        if self._registered:
            await self._runtime.unregister_tensor(self._arena.slab)
            self._registered = False

    def _require_worker_session_nonce(self) -> str:
        if self._worker_session_nonce is None:
            raise RuntimeError("Transfer Engine Worker session nonce is unavailable")
        return self._worker_session_nonce

    def _validate_output(self, output: torch.Tensor, batch: WorkerBatch) -> None:
        if output.shape != (batch.token_count, self._spec.hidden_dim):
            raise ValueError("Transfer Engine output has an unexpected shape")
        if output.dtype != self._spec.dtype:
            raise ValueError("Transfer Engine output dtype does not match the endpoint")
        if output.device != self._device or batch.hidden_states.device != self._device:
            raise ValueError("Transfer Engine output and batch must use the configured device")
        if not output.is_contiguous():
            raise ValueError("Transfer Engine output must be contiguous")


__all__ = ["TransferEngineWorkerTransport"]
