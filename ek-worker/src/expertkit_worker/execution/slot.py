"""Fixed Tensor storage and stream ordering for one active computation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol, override

import torch
from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.tracing import Tracer, TraceSpan
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
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
from expertkit_worker.device import AsyncWorkerDeviceRuntime, WorkerDeviceRuntime


def _trace_span(tracer: Tracer | None, name: str) -> Any:
    if tracer is None:
        return nullcontext(None)
    return tracer.start_as_current_span(name)


class ExecutionSlotFactory(Protocol):
    def __call__(
        self,
        spec: BatchBufferConfig,
        transport_buffers: WorkerBatchBuffers,
        *,
        enable_device_timing: bool = False,
    ) -> ExecutionSlot: ...


def _request_end_error(
    received: ReceivedBatch,
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


@dataclass(frozen=True, slots=True)
class _SlotTensors:
    hidden_states: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    output: torch.Tensor

    def valid_views(
        self,
        token_count: int,
    ) -> _SlotTensors:
        return _SlotTensors(
            hidden_states=self.hidden_states[:token_count],
            expert_ids=self.expert_ids[:token_count],
            routing_weights=self.routing_weights[:token_count],
            output=self.output[:token_count],
        )

    @property
    def device_bytes(self) -> int:
        """Return logical bytes reserved by fixed Backend input and output tensors."""
        return (
            self.hidden_states.numel() * self.hidden_states.element_size()
            + self.expert_ids.numel() * self.expert_ids.element_size()
            + self.routing_weights.numel() * self.routing_weights.element_size()
            + self.output.numel() * self.output.element_size()
        )


@dataclass(slots=True)
class ExecutionResult:
    """Hold one active slot until result communication has finished.

    Attributes:
        output: Tensor safe for the receiver to send. CUDA gRPC returns a
            pinned CPU view; CPU execution returns the fixed Backend output view.
        rejection: Cancellation or deadline result when no output should be sent.

    Note:
        Call :meth:`release` only after response communication no longer reads the
        output. This closes Backend completion state and makes the fixed slot
        reusable.
    """

    output: torch.Tensor | None
    rejection: TransportError | None
    _slot: ExecutionSlot = field(repr=False)
    _completion: BackendCompletion | None = field(repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def release(self) -> None:
        """Release Backend references and return the active slot exactly once."""

        if self._released:
            return
        self._released = True
        try:
            if self._completion is not None:
                self._completion.close()
        finally:
            self._slot._release_result(self)


class ExecutionSlot(ABC):
    """Own fixed input, output, staging, and CUDA ordering for one active batch."""

    def __init__(
        self,
        spec: BatchBufferConfig,
        transport_buffers: WorkerBatchBuffers,
        *,
        runtime: WorkerDeviceRuntime,
    ) -> None:
        self._spec = spec
        self._transport_buffers = transport_buffers

        try:
            if spec.device != runtime.device:
                raise ValueError("slot buffer device does not match its runtime")

            self._tensors: _SlotTensors | None = _SlotTensors(
                hidden_states=torch.empty(
                    (spec.max_batch_tokens, spec.hidden_dim),
                    dtype=spec.dtype,
                    device=spec.device,
                ),
                expert_ids=torch.empty(
                    (spec.max_batch_tokens, spec.top_k),
                    dtype=torch.int32,
                    device=spec.device,
                ),
                routing_weights=torch.empty(
                    (spec.max_batch_tokens, spec.top_k),
                    dtype=torch.float32,
                    device=spec.device,
                ),
                output=torch.empty(
                    (spec.max_batch_tokens, spec.hidden_dim),
                    dtype=spec.dtype,
                    device=spec.device,
                ),
            )
        except BaseException:
            transport_buffers.close()
            raise
        self._result: ExecutionResult | None = None
        self._busy = False
        self._closed = False
        self._state_lock = Lock()

    @abstractmethod
    def _execute_impl(
        self,
        received: ReceivedBatch,
        source: WorkerBatch,
        backend: ComputeBackend,
        tensors: _SlotTensors,
        *,
        clock: Callable[[], float],
        tracer: Tracer | None,
        batch_span: TraceSpan | None,
    ) -> ExecutionResult: ...

    @property
    def device(self) -> torch.device:
        """Return the device owned by this active slot."""
        return self._spec.device

    @property
    def device_bytes(self) -> int:
        """Return logical bytes reserved by fixed Backend input and output tensors."""
        return self._require_tensors().device_bytes

    @property
    def host_staging_bytes(self) -> int:
        """Return fixed Host bytes allocated by the selected Transport."""

        return self._transport_buffers.host_staging_bytes

    @property
    def busy(self) -> bool:
        """Return whether computation or communication still owns this slot."""

        with self._state_lock:
            return self._busy

    def execute(
        self,
        received: ReceivedBatch,
        backend: ComputeBackend,
        *,
        clock: Callable[[], float] = time.monotonic,
        tracer: Tracer | None = None,
        batch_span: TraceSpan | None = None,
    ) -> ExecutionResult:
        """Run one received batch in the current bounded execution thread.

        Returns:
            A result that keeps fixed storage and Backend resources active until its
            explicit release after response communication.

        Raises:
            BackendRequestError: The request can be rejected safely.
            BackendFatalError: Backend, copy, or device state is unsafe to continue.
            RuntimeError: The slot is closed or already active.
        """

        self._claim()
        try:
            source = received.batch
            initial_error = _request_end_error(received, clock)
            if initial_error is not None:
                received.release_input()
                return self._set_result(None, initial_error, None)

            try:
                return self._execute_impl(
                    received=received,
                    source=source,
                    backend=backend,
                    tensors=self._valid_views(source.token_count),
                    clock=clock,
                    tracer=tracer,
                    batch_span=batch_span,
                )
            except BackendRequestError:
                raise
            except BackendFatalError:
                raise
            except torch.OutOfMemoryError as error:
                raise BackendFatalError(
                    BackendFatalReason.DEVICE_OOM,
                    str(error),
                ) from error
            except ValueError as error:
                raise InvalidBackendInput(str(error)) from error
            except Exception as error:
                raise BackendFatalError(
                    BackendFatalReason.UNEXPECTED,
                    str(error),
                ) from error
        except BaseException:
            with self._state_lock:
                self._busy = False
            raise

    def close(self) -> None:
        """Release fixed resources after the slot becomes idle."""
        with self._state_lock:
            if self._closed:
                return
            if self._busy:
                raise RuntimeError("cannot close an active computation slot")
            self._closed = True
        self._transport_buffers.close()
        self._tensors = None

    def _copy_and_build_batch(
        self,
        source: WorkerBatch,
        received: ReceivedBatch,
        tensors: _SlotTensors,
    ) -> BackendBatch:
        hidden_states = tensors.hidden_states
        expert_ids = tensors.expert_ids
        routing_weights = tensors.routing_weights

        self._transport_buffers.copy_input(
            source,
            hidden_states,
            expert_ids,
            routing_weights,
        )
        layer_id = source.layer_id
        distinct_expert_ids = source.distinct_expert_ids
        received.release_input()
        return BackendBatch(
            layer_id=layer_id,
            hidden_states=hidden_states,
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
    ) -> _SlotTensors:
        if not 0 < token_count <= self._spec.max_batch_tokens:
            raise InvalidBackendInput("batch token count exceeds the active slot")
        return self._require_tensors().valid_views(token_count)

    def _set_result(
        self,
        output: torch.Tensor | None,
        rejection: TransportError | None,
        completion: BackendCompletion | None,
    ) -> ExecutionResult:
        result = ExecutionResult(output, rejection, self, completion)
        with self._state_lock:
            self._result = result
        return result

    def _release_result(self, result: ExecutionResult) -> None:
        with self._state_lock:
            if self._result is not result:
                raise RuntimeError("slot result does not own this active slot")
            self._result = None
            self._busy = False

    def _claim(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("active computation slot is closed")
            if self._busy:
                raise RuntimeError("active computation slot is already in use")
            self._busy = True

    def _require_tensors(self) -> _SlotTensors:
        tensors = self._tensors
        if tensors is None:
            raise RuntimeError("execution slot is closed")
        return tensors


class CpuExecutionSlot(ExecutionSlot):
    @override
    def _execute_impl(
        self,
        received: ReceivedBatch,
        source: WorkerBatch,
        backend: ComputeBackend,
        tensors: _SlotTensors,
        *,
        clock: Callable[[], float],
        tracer: Tracer | None,
        batch_span: TraceSpan | None,
    ) -> ExecutionResult:
        with _trace_span(tracer, "worker.input.prepare"):
            batch = self._copy_and_build_batch(source, received, tensors)

        rejection = _request_end_error(received, clock)
        if rejection is not None:
            return self._set_result(None, rejection, None)

        with _trace_span(tracer, "worker.backend.submit"):
            completion = self._submit(backend, batch, tensors.output)

        try:
            with _trace_span(tracer, "worker.backend.wait"):
                self._wait_completion(completion)

            rejection = _request_end_error(received, clock)

            with _trace_span(tracer, "worker.output.prepare"):
                response_output = (
                    None
                    if rejection is not None
                    else self._transport_buffers.copy_output(
                        tensors.output,
                        received.output_destination,
                    )
                )
            return self._set_result(response_output, rejection, completion)
        except BaseException:
            completion.close()
            raise


@dataclass(frozen=True, slots=True)
class _TimingEvents[EventT]:
    start: EventT
    after_input: EventT
    after_backend: EventT
    after_output: EventT


@dataclass(frozen=True, slots=True)
class _AsyncSlotResources[StreamT, EventT]:
    stream: StreamT
    timing_events: _TimingEvents[EventT] | None = None


class AsyncExecutionSlot[StreamT, EventT](ExecutionSlot):
    def __init__(
        self,
        spec: BatchBufferConfig,
        transport_buffers: WorkerBatchBuffers,
        *,
        enable_device_timing: bool,
        runtime: AsyncWorkerDeviceRuntime[StreamT, EventT],
    ) -> None:
        super().__init__(spec, transport_buffers, runtime=runtime)
        try:
            with runtime.device_context():
                self._resources: _AsyncSlotResources[StreamT, EventT] | None
                self._resources = _AsyncSlotResources(
                    stream=runtime.create_stream(),
                    timing_events=self._create_timing_events(runtime, enable_device_timing),
                )
        except BaseException:
            super().close()
            raise

        self._runtime = runtime

    @override
    def close(self) -> None:
        super().close()
        self._resources = None

    @override
    def _execute_impl(
        self,
        received: ReceivedBatch,
        source: WorkerBatch,
        backend: ComputeBackend,
        tensors: _SlotTensors,
        *,
        clock: Callable[[], float],
        tracer: Tracer | None,
        batch_span: TraceSpan | None,
    ) -> ExecutionResult:
        resources = self._require_async_resources()
        timing_events = (
            resources.timing_events
            if batch_span is not None and batch_span.is_recording()
            else None
        )
        stream = resources.stream
        runtime = self._runtime

        completion: BackendCompletion | None = None
        rejection: TransportError | None = None
        response_output: torch.Tensor | None = None

        try:
            with runtime.device_context(), runtime.stream_context(stream):
                if timing_events is not None:
                    runtime.record_event(timing_events.start, stream)

                with _trace_span(tracer, "worker.input.prepare"):
                    batch = self._copy_and_build_batch(source, received, tensors)
                if timing_events is not None:
                    runtime.record_event(timing_events.after_input, stream)

                # Only submit batch to backend if there is no rejection
                rejection = _request_end_error(received, clock)
                if rejection is None:
                    with _trace_span(tracer, "worker.backend.submit"):
                        completion = self._submit(backend, batch, tensors.output)

                if timing_events is not None:
                    runtime.record_event(timing_events.after_backend, stream)

                if completion is not None:
                    rejection = _request_end_error(received, clock)
                    if rejection is None:
                        with _trace_span(tracer, "worker.output.prepare"):
                            response_output = self._transport_buffers.copy_output(
                                tensors.output,
                                received.output_destination,
                            )
                if timing_events is not None:
                    runtime.record_event(timing_events.after_output, stream)
            try:
                with _trace_span(tracer, "worker.device.wait"):
                    runtime.synchronize_stream(stream)
            except torch.OutOfMemoryError as error:
                raise BackendFatalError(BackendFatalReason.DEVICE_OOM, str(error)) from error
            except Exception as error:
                raise BackendFatalError(
                    BackendFatalReason.ASYNC_EXECUTION,
                    str(error),
                ) from error

            if timing_events is not None and batch_span is not None:
                self._emit_device_timings(timing_events, batch_span)

            if completion is not None:
                with _trace_span(tracer, "worker.backend.wait"):
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

    def _emit_device_timings(
        self, timing_events: _TimingEvents[EventT], batch_span: TraceSpan
    ) -> None:
        runtime = self._runtime
        input_ms = runtime.elapsed_time_ms(
            timing_events.start,
            timing_events.after_input,
        )
        backend_ms = runtime.elapsed_time_ms(
            timing_events.after_input,
            timing_events.after_backend,
        )
        output_ms = runtime.elapsed_time_ms(
            timing_events.after_backend,
            timing_events.after_output,
        )
        batch_span.set_attribute("expertkit.device.input_stage_ms", input_ms)
        batch_span.set_attribute("expertkit.device.backend_stage_ms", backend_ms)
        batch_span.set_attribute("expertkit.device.output_stage_ms", output_ms)
        batch_span.set_attribute(
            "expertkit.device.total_stage_ms",
            input_ms + backend_ms + output_ms,
        )

    def _require_async_resources(self) -> _AsyncSlotResources[StreamT, EventT]:
        resources = self._resources
        if resources is None:
            raise RuntimeError("execution slot is closed")
        return resources

    @staticmethod
    def _create_timing_events(
        runtime: AsyncWorkerDeviceRuntime[StreamT, EventT],
        enable_device_timing: bool,
    ) -> _TimingEvents[EventT] | None:
        return (
            _TimingEvents(
                start=runtime.create_event(enable_timing=True),
                after_input=runtime.create_event(enable_timing=True),
                after_backend=runtime.create_event(enable_timing=True),
                after_output=runtime.create_event(enable_timing=True),
            )
            if enable_device_timing
            else None
        )
