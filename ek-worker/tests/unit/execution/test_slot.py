"""Tests for fixed active computation slots."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import ClassVar

import pytest
import torch
from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.tracing import TraceAttribute, TraceContext, TraceSpan
from expertkit_transport.transports.base import BatchBufferConfig, ReceivedBatch
from expertkit_transport.transports.grpc.worker_buffers import GrpcWorkerBatchBuffers

from expertkit_worker.backends import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendFatalError,
    BackendFatalReason,
    BackendResourceEstimate,
    ComputeBackend,
)
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime
from expertkit_worker.execution import AsyncExecutionSlot, CpuExecutionSlot, ExecutionSlot


class FakeReceivedBatch(ReceivedBatch):
    """Expose controlled cancellation and input-lifetime state to a slot."""

    def __init__(self, batch: WorkerBatch, *, deadline: float = math.inf) -> None:
        self._batch: WorkerBatch | None = batch
        self._deadline = deadline
        self.is_cancelled = False
        self.input_released = False

    @property
    def trace_context(self) -> object | None:
        return None

    @property
    def batch(self) -> WorkerBatch:
        if self._batch is None:
            raise RuntimeError("input released")
        return self._batch

    @property
    def monotonic_deadline(self) -> float:
        return self._deadline

    @property
    def cancelled(self) -> bool:
        return self.is_cancelled

    @property
    def output_destination(self) -> torch.Tensor | None:
        return None

    def release_input(self) -> None:
        self.input_released = True
        self._batch = None

    async def complete(self, partial_output: torch.Tensor) -> None:
        raise AssertionError("slot tests do not send responses")

    async def reject(self, error: TransportError) -> None:
        raise AssertionError("slot tests do not send responses")


class RecordingSpan:
    def __init__(self, *, recording: bool = True) -> None:
        self.attributes: dict[str, TraceAttribute] = {}
        self.recording = recording

    def set_attribute(self, key: str, value: TraceAttribute) -> None:
        self.attributes[key] = value

    def is_recording(self) -> bool:
        return self.recording

    def end(self) -> None:
        pass


class RecordingTracer:
    def __init__(self) -> None:
        self.names: list[str] = []

    def current_span_is_recording(self) -> bool:
        return True

    def capture_context(self) -> TraceContext:
        return object()

    def start_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> TraceSpan:
        self.names.append(name)
        return RecordingSpan()

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> Iterator[TraceSpan]:
        self.names.append(name)
        yield RecordingSpan()


class TrackingCompletion(BackendCompletion):
    """Track wait and close calls made by Worker execution."""

    def __init__(self, retained: Iterable[object] = ()) -> None:
        self.retained = list(retained)
        self.waited = False
        self.closed = False

    def wait_host(self) -> None:
        self.waited = True

    def close(self) -> None:
        self.closed = True
        self.retained.clear()


class DoublingBackend(ComputeBackend):
    """Write a deterministic result and record fixed Tensor addresses."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_pointers: list[int] = []
        self.output_pointers: list[int] = []
        self.streams: list[int | None] = []
        self.completions: list[TrackingCompletion] = []
        self.cancel_during_submit: FakeReceivedBatch | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_dynamic_tokens=True,
            supports_concurrent_batches=True,
        )

    def estimate_resources(self, max_batch_tokens: int) -> BackendResourceEstimate:
        return BackendResourceEstimate(temporary_bytes_per_active_batch=0)

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        self.calls += 1
        self.input_pointers.append(batch.hidden_states.data_ptr())
        self.output_pointers.append(prepared_output.data_ptr())
        self.streams.append(
            torch.cuda.current_stream(batch.hidden_states.device).cuda_stream
            if batch.hidden_states.device.type == "cuda"
            else None
        )
        torch.mul(batch.hidden_states, 2, out=prepared_output)
        if self.cancel_during_submit is not None:
            self.cancel_during_submit.is_cancelled = True
        completion = TrackingCompletion((batch,))
        self.completions.append(completion)
        return completion


def worker_batch(dtype: torch.dtype = torch.float32) -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype),
        token_indices=None,
        expert_ids=torch.tensor([[1, -1], [3, 1]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0.0], [0.75, 0.25]], dtype=torch.float32),
        distinct_expert_ids=(1, 3),
    )


def make_slot(
    device: str,
    dtype: torch.dtype = torch.float32,
    *,
    enable_device_timing: bool = False,
) -> ExecutionSlot:
    spec = BatchBufferConfig(
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=dtype,
        device=torch.device(device),
    )
    buffers = GrpcWorkerBatchBuffers(spec)
    if spec.device.type == "cpu":
        return CpuExecutionSlot(
            spec,
            buffers,
            runtime=CpuWorkerRuntime(spec.device),
        )
    return AsyncExecutionSlot(
        spec,
        buffers,
        enable_device_timing=enable_device_timing,
        runtime=CudaWorkerRuntime(spec.device),
    )


def test_cpu_slot_reuses_fixed_inputs_and_output() -> None:
    slot = make_slot("cpu")
    backend = DoublingBackend()

    first = FakeReceivedBatch(worker_batch())
    expected = first.batch.hidden_states * 2
    first_result = slot.execute(first, backend)
    torch.testing.assert_close(first_result.output, expected)
    assert first.input_released is True
    assert backend.completions[0].waited is True
    assert slot.busy is True
    with pytest.raises(RuntimeError, match="already in use"):
        slot.execute(FakeReceivedBatch(worker_batch()), backend)
    first_result.release()
    assert backend.completions[0].closed is True

    second_result = slot.execute(FakeReceivedBatch(worker_batch()), backend)
    assert backend.input_pointers[0] == backend.input_pointers[1]
    assert backend.output_pointers[0] == backend.output_pointers[1]
    second_result.release()
    assert slot.device_bytes == 160
    assert slot.host_staging_bytes == 0
    slot.close()


@pytest.mark.parametrize(
    ("cancelled", "deadline", "code"),
    [
        (True, math.inf, TransportErrorCode.CANCELLED),
        (False, 10.0, TransportErrorCode.DEADLINE_EXCEEDED),
    ],
)
def test_slot_rejects_ended_request_before_backend_submission(
    cancelled: bool,
    deadline: float,
    code: TransportErrorCode,
) -> None:
    slot = make_slot("cpu")
    backend = DoublingBackend()
    received = FakeReceivedBatch(worker_batch(), deadline=deadline)
    received.is_cancelled = cancelled

    result = slot.execute(received, backend, clock=lambda: 10.0)

    assert result.output is None
    assert result.rejection is not None
    assert result.rejection.code is code
    assert backend.calls == 0
    assert received.input_released is True
    result.release()
    slot.close()


def test_slot_tracks_submitted_work_after_cancellation() -> None:
    slot = make_slot("cpu")
    backend = DoublingBackend()
    received = FakeReceivedBatch(worker_batch())
    backend.cancel_during_submit = received

    result = slot.execute(received, backend)

    assert result.output is None
    assert result.rejection is not None
    assert result.rejection.code is TransportErrorCode.CANCELLED
    assert backend.completions[0].waited is True
    assert backend.completions[0].closed is False
    result.release()
    assert backend.completions[0].closed is True
    slot.close()


def test_slot_maps_unclassified_backend_exception_to_fatal() -> None:
    class BrokenBackend(DoublingBackend):
        def submit(
            self,
            batch: BackendBatch,
            prepared_output: torch.Tensor,
        ) -> BackendCompletion:
            raise KeyError("broken state")

    slot = make_slot("cpu")

    with pytest.raises(BackendFatalError) as caught:
        slot.execute(FakeReceivedBatch(worker_batch()), BrokenBackend())

    assert caught.value.reason is BackendFatalReason.UNEXPECTED
    assert slot.busy is False
    slot.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_slot_uses_one_stream_and_reuses_pinned_output() -> None:
    slot = make_slot("cuda:0", torch.float16, enable_device_timing=True)
    backend = DoublingBackend()
    tracer = RecordingTracer()
    batch_span = RecordingSpan()

    first = slot.execute(
        FakeReceivedBatch(worker_batch(torch.float16)),
        backend,
        tracer=tracer,
        batch_span=batch_span,
    )
    assert first.output is not None
    output_pointer = first.output.data_ptr()
    assert first.output.is_pinned()
    torch.testing.assert_close(
        first.output,
        torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.float16),
    )
    first.release()

    unsampled_span = RecordingSpan(recording=False)
    second = slot.execute(
        FakeReceivedBatch(worker_batch(torch.float16)),
        backend,
        tracer=tracer,
        batch_span=unsampled_span,
    )
    assert second.output is not None
    assert second.output.data_ptr() == output_pointer
    assert backend.input_pointers[0] == backend.input_pointers[1]
    assert backend.output_pointers[0] == backend.output_pointers[1]
    assert backend.streams[0] == backend.streams[1]
    assert backend.streams[0] is not None
    for name in (
        "expertkit.device.input_stage_ms",
        "expertkit.device.backend_stage_ms",
        "expertkit.device.output_stage_ms",
        "expertkit.device.total_stage_ms",
    ):
        value = batch_span.attributes[name]
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value >= 0
    assert "worker.device.wait" in tracer.names
    assert not any(name.startswith("expertkit.device.") for name in unsampled_span.attributes)
    second.release()
    assert slot.device_bytes == 112
    assert slot.host_staging_bytes == 112
    slot.close()


class _FakeStream:
    pass


class _FakeEvent:
    def __init__(self, *, enable_timing: bool) -> None:
        self.enable_timing = enable_timing
        self.sequence = 0
        self.synchronized = False


class _CompletedWork:
    def wait_host(self) -> None:
        pass


class _FakeAsyncRuntime:
    def __init__(self, *, fail_stream_creation: bool = False) -> None:
        self.device = torch.device("cpu")
        self.fail_stream_creation = fail_stream_creation
        self.current_device_set = False
        self.stream_synchronized = False
        self._sequence = 0

    @contextmanager
    def device_context(self) -> Iterator[None]:
        yield

    def memory_info(self) -> tuple[int, int]:
        return 1, 1

    def capture_current_work(self) -> _CompletedWork:
        return _CompletedWork()

    def create_stream(self, *, priority: int = 0) -> _FakeStream:
        assert priority == 0
        if self.fail_stream_creation:
            raise RuntimeError("stream creation failed")
        return _FakeStream()

    def current_stream(self) -> _FakeStream:
        return _FakeStream()

    def synchronize_stream(self, stream: _FakeStream) -> None:
        assert isinstance(stream, _FakeStream)
        self.stream_synchronized = True

    @contextmanager
    def stream_context(self, stream: _FakeStream) -> Iterator[None]:
        assert isinstance(stream, _FakeStream)
        yield

    def create_event(self, *, enable_timing: bool = False) -> _FakeEvent:
        return _FakeEvent(enable_timing=enable_timing)

    def record_event(self, event: _FakeEvent, stream: _FakeStream) -> None:
        assert isinstance(stream, _FakeStream)
        self._sequence += 1
        event.sequence = self._sequence

    def wait_event(self, stream: _FakeStream, event: _FakeEvent) -> None:
        assert isinstance(stream, _FakeStream)
        assert isinstance(event, _FakeEvent)

    def synchronize_event(self, event: _FakeEvent) -> None:
        event.synchronized = True

    def event_done(self, event: _FakeEvent) -> bool:
        return event.synchronized

    def set_current_device(self) -> None:
        self.current_device_set = True

    def synchronize_device(self) -> None:
        pass

    def elapsed_time_ms(self, start: _FakeEvent, end: _FakeEvent) -> float:
        return float(end.sequence - start.sequence)


def _cpu_spec() -> BatchBufferConfig:
    return BatchBufferConfig(
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_async_slot_uses_injected_stream_events_and_device_timings() -> None:
    spec = _cpu_spec()
    runtime = _FakeAsyncRuntime()
    slot = AsyncExecutionSlot(
        spec,
        GrpcWorkerBatchBuffers(spec),
        enable_device_timing=True,
        runtime=runtime,
    )
    span = RecordingSpan()

    result = slot.execute(
        FakeReceivedBatch(worker_batch()),
        DoublingBackend(),
        batch_span=span,
    )

    assert runtime.stream_synchronized is True
    assert result.output is not None
    torch.testing.assert_close(result.output, worker_batch().hidden_states * 2)
    assert span.attributes == {
        "expertkit.device.input_stage_ms": 1.0,
        "expertkit.device.backend_stage_ms": 1.0,
        "expertkit.device.output_stage_ms": 1.0,
        "expertkit.device.total_stage_ms": 3.0,
    }
    result.release()
    slot.close()


class _InspectableAsyncSlot(AsyncExecutionSlot[_FakeStream, _FakeEvent]):
    latest: ClassVar[_InspectableAsyncSlot | None] = None

    def __new__(cls, *_args: object, **_kwargs: object) -> _InspectableAsyncSlot:
        instance = super().__new__(cls)
        cls.latest = instance
        return instance


def test_async_slot_releases_base_resources_when_stream_creation_fails() -> None:
    spec = _cpu_spec()
    runtime = _FakeAsyncRuntime(fail_stream_creation=True)

    with pytest.raises(RuntimeError, match="stream creation failed"):
        _InspectableAsyncSlot(
            spec,
            GrpcWorkerBatchBuffers(spec),
            enable_device_timing=False,
            runtime=runtime,
        )

    slot = _InspectableAsyncSlot.latest
    assert slot is not None
    assert slot._tensors is None
