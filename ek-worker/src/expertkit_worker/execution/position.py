"""Fixed Tensor storage and stream ordering for one active computation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock

import torch
from expertkit_transport.contracts import (
    ReceivedWorkerBatch,
    TransportError,
    TransportErrorCode,
    WorkerBatch,
    WorkerPositionBuffers,
    WorkerPositionSpec,
)

from expertkit_worker.backends import (
    BackendBatch,
    BackendCompletion,
    BackendFatalError,
    BackendFatalReason,
    BackendRequestError,
    ComputeBackend,
    InvalidBackendInput,
)


def _request_end_error(
    received: ReceivedWorkerBatch,
    clock: Callable[[], float],
) -> TransportError | None:
    if received.cancelled:
        return TransportError(
            TransportErrorCode.CANCELLED,
            retryable=False,
            diagnostic="computation caller cancelled before response completion",
        )
    if clock() >= received.monotonic_deadline:
        return TransportError(
            TransportErrorCode.DEADLINE_EXCEEDED,
            retryable=False,
            diagnostic="computation deadline expired before response completion",
        )
    return None


@dataclass(slots=True)
class PositionResult:
    """Hold one active position until result communication has finished.

    Attributes:
        output: Tensor safe for the receiver adapter to send. CUDA gRPC returns a
            pinned CPU view; CPU execution returns the fixed Backend output view.
        rejection: Cancellation or deadline result when no output should be sent.

    Note:
        Call :meth:`release` only after response communication no longer reads the
        output. This closes Backend completion state and makes the fixed position
        reusable.
    """

    output: torch.Tensor | None
    rejection: TransportError | None
    _position: ActivePosition = field(repr=False)
    _completion: BackendCompletion | None = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def release(self) -> None:
        """Release Backend references and return the active position exactly once."""

        if self._released:
            return
        self._released = True
        try:
            if self._completion is not None:
                self._completion.close()
        finally:
            self._position._release_result(self)


class ActivePosition:
    """Own fixed input, output, staging, and CUDA ordering for one active batch."""

    def __init__(
        self,
        spec: WorkerPositionSpec,
        transport_buffers: WorkerPositionBuffers,
    ) -> None:
        self._spec = spec
        self._transport_buffers = transport_buffers
        try:
            self._hidden_states: torch.Tensor | None = torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                device=spec.device,
            )
            self._expert_ids: torch.Tensor | None = torch.empty(
                (spec.max_batch_tokens, spec.top_k),
                dtype=torch.int32,
                device=spec.device,
            )
            self._routing_weights: torch.Tensor | None = torch.empty(
                (spec.max_batch_tokens, spec.top_k),
                dtype=torch.float32,
                device=spec.device,
            )
            self._partial_output: torch.Tensor | None = torch.empty(
                (spec.max_batch_tokens, spec.hidden_dim),
                dtype=spec.dtype,
                device=spec.device,
            )
            if spec.device.type == "cuda":
                with torch.cuda.device(spec.device):
                    self._stream: torch.cuda.Stream | None = torch.cuda.Stream(device=spec.device)
                    self._event: torch.cuda.Event | None = torch.cuda.Event()
            else:
                self._stream = None
                self._event = None
        except BaseException:
            transport_buffers.close()
            raise
        self._result: PositionResult | None = None
        self._busy = False
        self._closed = False
        self._state_lock = Lock()

    @property
    def device_bytes(self) -> int:
        """Return logical bytes reserved by fixed Backend input and output tensors."""

        return sum(
            tensor.numel() * tensor.element_size() for tensor in self._require_device_tensors()
        )

    @property
    def host_staging_bytes(self) -> int:
        """Return fixed Host bytes allocated by the selected Transport adapter."""

        return self._transport_buffers.host_staging_bytes

    @property
    def busy(self) -> bool:
        """Return whether computation or communication still owns this position."""

        with self._state_lock:
            return self._busy

    def execute(
        self,
        received: ReceivedWorkerBatch,
        backend: ComputeBackend,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> PositionResult:
        """Run one received batch in the current bounded execution thread.

        Returns:
            A result that keeps fixed storage and Backend resources active until its
            explicit release after response communication.

        Raises:
            BackendRequestError: The request can be rejected safely.
            BackendFatalError: Backend, copy, or device state is unsafe to continue.
            RuntimeError: The position is closed or already active.
        """

        self._claim()
        try:
            source = received.batch
            initial_error = _request_end_error(received, clock)
            if initial_error is not None:
                received.release_input()
                return self._set_result(None, initial_error, None)

            token_count = source.token_count
            hidden, expert_ids, routing_weights, output = self._valid_views(token_count)
            try:
                if self._spec.device.type == "cuda":
                    return self._execute_cuda(
                        received,
                        source,
                        backend,
                        hidden,
                        expert_ids,
                        routing_weights,
                        output,
                        clock,
                    )
                return self._execute_cpu(
                    received,
                    source,
                    backend,
                    hidden,
                    expert_ids,
                    routing_weights,
                    output,
                    clock,
                )
            except BackendRequestError:
                raise
            except BackendFatalError:
                raise
            except torch.OutOfMemoryError as error:
                raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
            except ValueError as error:
                raise InvalidBackendInput(str(error)) from error
            except Exception as error:
                raise BackendFatalError(BackendFatalReason.UNEXPECTED, str(error)) from error
        except BaseException:
            with self._state_lock:
                self._busy = False
            raise

    def close(self) -> None:
        """Release fixed resources after the position becomes idle."""

        with self._state_lock:
            if self._closed:
                return
            if self._busy:
                raise RuntimeError("cannot close an active computation position")
            self._closed = True
        self._transport_buffers.close()
        self._hidden_states = None
        self._expert_ids = None
        self._routing_weights = None
        self._partial_output = None
        self._stream = None
        self._event = None

    def _execute_cpu(
        self,
        received: ReceivedWorkerBatch,
        source: WorkerBatch,
        backend: ComputeBackend,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
        clock: Callable[[], float],
    ) -> PositionResult:
        batch = self._copy_and_build_batch(
            received,
            source,
            hidden,
            expert_ids,
            routing_weights,
        )
        rejection = _request_end_error(received, clock)
        if rejection is not None:
            return self._set_result(None, rejection, None)
        completion = self._submit(backend, batch, output)
        try:
            self._wait_completion(completion)
            rejection = _request_end_error(received, clock)
            response_output = (
                None if rejection is not None else self._transport_buffers.copy_output(output)
            )
            return self._set_result(response_output, rejection, completion)
        except BaseException:
            completion.close()
            raise

    def _execute_cuda(
        self,
        received: ReceivedWorkerBatch,
        source: WorkerBatch,
        backend: ComputeBackend,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
        clock: Callable[[], float],
    ) -> PositionResult:
        stream = self._require_stream()
        event = self._require_event()
        completion: BackendCompletion | None = None
        response_output: torch.Tensor | None = None
        rejection: TransportError | None = None
        try:
            with torch.cuda.device(self._spec.device), torch.cuda.stream(stream):
                batch = self._copy_and_build_batch(
                    received,
                    source,
                    hidden,
                    expert_ids,
                    routing_weights,
                )
                rejection = _request_end_error(received, clock)
                if rejection is None:
                    completion = self._submit(backend, batch, output)
                    rejection = _request_end_error(received, clock)
                    if rejection is None:
                        response_output = self._transport_buffers.copy_output(output)
                event.record(stream)

            try:
                event.synchronize()
            except torch.OutOfMemoryError as error:
                raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
            except Exception as error:
                raise BackendFatalError(BackendFatalReason.ASYNC_EXECUTION, str(error)) from error
            if completion is not None:
                self._wait_completion(completion)
            final_rejection = _request_end_error(received, clock)
            if final_rejection is not None:
                rejection = final_rejection
                response_output = None
            return self._set_result(response_output, rejection, completion)
        except BaseException:
            if completion is not None:
                completion.close()
            raise

    def _copy_and_build_batch(
        self,
        received: ReceivedWorkerBatch,
        source: WorkerBatch,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> BackendBatch:
        self._transport_buffers.copy_input(source, hidden, expert_ids, routing_weights)
        layer_id = source.layer_id
        distinct_expert_ids = source.distinct_expert_ids
        received.release_input()
        return BackendBatch(
            layer_id=layer_id,
            hidden_states=hidden,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            distinct_expert_ids=distinct_expert_ids,
        )

    @staticmethod
    def _submit(
        backend: ComputeBackend,
        batch: BackendBatch,
        output: torch.Tensor,
    ) -> BackendCompletion:
        try:
            return backend.submit(batch, output)
        except BackendRequestError:
            raise
        except BackendFatalError:
            raise
        except torch.OutOfMemoryError as error:
            raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
        except Exception as error:
            raise BackendFatalError(BackendFatalReason.UNEXPECTED, str(error)) from error

    @staticmethod
    def _wait_completion(completion: BackendCompletion) -> None:
        try:
            completion.wait_host()
        except BackendRequestError:
            raise
        except BackendFatalError:
            raise
        except torch.OutOfMemoryError as error:
            raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
        except Exception as error:
            raise BackendFatalError(BackendFatalReason.UNEXPECTED, str(error)) from error

    def _valid_views(
        self,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 < token_count <= self._spec.max_batch_tokens:
            raise InvalidBackendInput("batch token count exceeds the active position")
        hidden, expert_ids, routing_weights, output = self._require_device_tensors()
        return (
            hidden[:token_count],
            expert_ids[:token_count],
            routing_weights[:token_count],
            output[:token_count],
        )

    def _set_result(
        self,
        output: torch.Tensor | None,
        rejection: TransportError | None,
        completion: BackendCompletion | None,
    ) -> PositionResult:
        result = PositionResult(output, rejection, self, completion)
        with self._state_lock:
            self._result = result
        return result

    def _release_result(self, result: PositionResult) -> None:
        with self._state_lock:
            if self._result is not result:
                raise RuntimeError("position result does not own this active position")
            self._result = None
            self._busy = False

    def _claim(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("active computation position is closed")
            if self._busy:
                raise RuntimeError("active computation position is already in use")
            self._busy = True

    def _require_device_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = (
            self._hidden_states,
            self._expert_ids,
            self._routing_weights,
            self._partial_output,
        )
        if any(tensor is None for tensor in tensors):
            raise RuntimeError("active computation position is closed")
        return tensors  # type: ignore[return-value]

    def _require_stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            raise RuntimeError("CUDA stream is not available for this position")
        return self._stream

    def _require_event(self) -> torch.cuda.Event:
        if self._event is None:
            raise RuntimeError("CUDA event is not available for this position")
        return self._event
