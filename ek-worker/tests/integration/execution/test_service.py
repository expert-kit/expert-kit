"""Integration tests for bounded Worker execution over real gRPC calls."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable

import pytest
import torch
from expertkit_transport.adapters.grpc import (
    GrpcBatchSpec,
    GrpcWorkerServer,
    GrpcWorkerTransport,
)
from expertkit_transport.contracts import (
    OutputSpec,
    TransportError,
    TransportErrorCode,
    WorkerBatch,
    WorkerPositionSpec,
)

from expertkit_worker.backends import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendFatalError,
    BackendFatalReason,
    BackendResourceEstimate,
    BackendWeightUnavailable,
    CompletedSubmission,
    ComputeBackend,
)
from expertkit_worker.execution import WorkerExecution
from expertkit_worker.observability.api import WorkerMetrics


class RecordingMetrics:
    """Record execution observations without an exporter dependency."""

    def __init__(self) -> None:
        self.active = 0
        self.finished: list[tuple[bool, float]] = []
        self.rejections: list[str] = []

    def batch_started(self) -> None:
        self.active += 1

    def batch_finished(self, *, fatal: bool, duration_seconds: float) -> None:
        self.active -= 1
        self.finished.append((fatal, duration_seconds))

    def batch_rejected(self, reason: str) -> None:
        self.rejections.append(reason)

    def pending_batches_changed(self, count: int) -> None:
        raise AssertionError("execution must not report Transport waiting counts")

    def weight_source_result(self, source: str, *, success: bool) -> None:
        raise AssertionError("execution must not report weight-source outcomes")

    def expert_state_changed(self, state: str) -> None:
        raise AssertionError("execution must not report expert state")

    def device_weight_bytes_changed(self, byte_count: int) -> None:
        raise AssertionError("execution must not report ready-weight bytes")


class TestBackend(ComputeBackend):
    """Run a small deterministic computation in Worker execution threads."""

    __test__ = False

    def __init__(self) -> None:
        self.thread_names: list[str] = []
        self.error: Exception | None = None

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
        self.thread_names.append(threading.current_thread().name)
        if self.error is not None:
            raise self.error
        torch.mul(batch.hidden_states, 3, out=prepared_output)
        return CompletedSubmission()


class BlockingBackend(TestBackend):
    """Hold active execution threads to expose execution and pending bounds."""

    def __init__(self, expected_active: int) -> None:
        super().__init__()
        self._expected_active = expected_active
        self._lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()

    def submit(
        self,
        batch: BackendBatch,
        prepared_output: torch.Tensor,
    ) -> BackendCompletion:
        with self._lock:
            self.thread_names.append(threading.current_thread().name)
            if len(self.thread_names) == self._expected_active:
                self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test did not release blocked Backend calls")
        torch.mul(batch.hidden_states, 3, out=prepared_output)
        return CompletedSubmission()


def batch_spec() -> GrpcBatchSpec:
    return GrpcBatchSpec(
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float32,
    )


def position_spec() -> WorkerPositionSpec:
    return WorkerPositionSpec(
        max_batch_tokens=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float32,
        device="cpu",
    )


def worker_batch() -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        token_indices=None,
        expert_ids=torch.tensor([[1, -1], [3, 1]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0.0], [0.75, 0.25]], dtype=torch.float32),
        distinct_expert_ids=(1, 3),
    )


def run(coroutine: Awaitable[None]) -> None:
    asyncio.run(coroutine)


async def start_stack(
    backend: ComputeBackend,
    *,
    active: int = 1,
    pending: int = 1,
    metrics: WorkerMetrics | None = None,
) -> tuple[GrpcWorkerServer, GrpcWorkerTransport, WorkerExecution]:
    server = GrpcWorkerServer(
        "127.0.0.1:0",
        batch_spec(),
        max_active_batches=active,
        max_pending_batches=pending,
    )
    execution = WorkerExecution(
        server,
        backend,
        instance_id=7,
        position_spec=position_spec(),
        active_positions=active,
        metrics=metrics,
    )
    await server.start()
    await execution.start()
    client = GrpcWorkerTransport(
        f"127.0.0.1:{server.bound_port}",
        batch_spec(),
        max_in_flight=active + pending,
    )
    await client.start()
    return server, client, execution


async def close_stack(
    client: GrpcWorkerTransport,
    execution: WorkerExecution,
) -> None:
    async with asyncio.timeout(5):
        try:
            await client.close()
        finally:
            await execution.close()


def test_execution_runs_backend_outside_event_loop_and_returns_output() -> None:
    async def scenario() -> None:
        backend = TestBackend()
        _server, client, execution = await start_stack(backend)
        output = client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu"))
        try:
            async with asyncio.timeout(2):
                await client.submit(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )

            torch.testing.assert_close(
                output.tensor[:2],
                torch.tensor([[3, 6, 9], [12, 15, 18]], dtype=torch.float32),
            )
            assert backend.thread_names[0].startswith("expertkit-worker-execution")
            assert execution.fixed_device_bytes == 160
            assert execution.fixed_host_staging_bytes == 0
        finally:
            client.output_buffers.release(output)
            await close_stack(client, execution)

    run(scenario())


def test_execution_maps_ready_weight_failure_without_stopping_worker() -> None:
    async def scenario() -> None:
        backend = TestBackend()
        backend.error = BackendWeightUnavailable((3,))
        metrics = RecordingMetrics()
        _server, client, execution = await start_stack(backend, metrics=metrics)
        output = client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu"))
        try:
            with pytest.raises(TransportError) as caught:
                async with asyncio.timeout(2):
                    await client.submit(
                        worker_batch(),
                        output,
                        monotonic_deadline=float("inf"),
                    )

            assert caught.value.code is TransportErrorCode.EXPERT_NOT_READY
            assert caught.value.unavailable_expert_ids == (3,)
            assert metrics.active == 0
            assert metrics.rejections == [TransportErrorCode.EXPERT_NOT_READY.value]
            assert len(metrics.finished) == 1
            assert metrics.finished[0][0] is False
            assert metrics.finished[0][1] >= 0
        finally:
            client.output_buffers.release(output)
            await close_stack(client, execution)

    run(scenario())


def test_execution_limits_active_work_to_fixed_positions() -> None:
    async def scenario() -> None:
        backend = BlockingBackend(expected_active=2)
        server, client, execution = await start_stack(backend, active=2, pending=2)
        outputs = [
            client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu")) for _ in range(4)
        ]
        submissions = [
            asyncio.create_task(
                client.submit(
                    worker_batch(),
                    output,
                    monotonic_deadline=float("inf"),
                )
            )
            for output in outputs
        ]
        try:
            started = await asyncio.to_thread(backend.started.wait, 2)
            assert started is True
            async with asyncio.timeout(2):
                await server._wait_pending_count(2)
            assert len(backend.thread_names) == 2
            assert server.active_count == 2
            assert server.pending_count == 2
            backend.release.set()
            await asyncio.gather(*submissions)

            assert len(backend.thread_names) == 4
        finally:
            backend.release.set()
            await asyncio.gather(*submissions, return_exceptions=True)
            for output in outputs:
                client.output_buffers.release(output)
            await close_stack(client, execution)

    run(scenario())


def test_cancelled_response_does_not_cancel_active_computation() -> None:
    async def scenario() -> None:
        backend = BlockingBackend(expected_active=1)
        server, client, execution = await start_stack(backend)
        first_output = client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu"))
        second_output = client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu"))
        first = asyncio.create_task(
            client.submit(
                worker_batch(),
                first_output,
                monotonic_deadline=float("inf"),
            )
        )
        try:
            assert await asyncio.to_thread(backend.started.wait, 2) is True
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            backend.release.set()
            async with asyncio.timeout(2):
                await server.wait_all_idle(monotonic_deadline=float("inf"))
                await client.submit(
                    worker_batch(),
                    second_output,
                    monotonic_deadline=float("inf"),
                )

            torch.testing.assert_close(
                second_output.tensor[:2],
                torch.tensor([[3, 6, 9], [12, 15, 18]], dtype=torch.float32),
            )
            assert server.active_count == 0
            assert server.pending_count == 0
        finally:
            backend.release.set()
            await asyncio.gather(first, return_exceptions=True)
            client.output_buffers.release(first_output)
            client.output_buffers.release(second_output)
            await close_stack(client, execution)

    run(scenario())


def test_fatal_backend_error_stops_execution() -> None:
    async def scenario() -> None:
        backend = TestBackend()
        backend.error = BackendFatalError(BackendFatalReason.DEVICE_OOM, "device exhausted")
        _server, client, execution = await start_stack(backend)
        output = client.output_buffers.prepare(OutputSpec(4, 3, torch.float32, "cpu"))
        try:
            with pytest.raises(TransportError) as caught:
                async with asyncio.timeout(2):
                    await client.submit(
                        worker_batch(),
                        output,
                        monotonic_deadline=float("inf"),
                    )
            with pytest.raises(BackendFatalError) as fatal:
                async with asyncio.timeout(2):
                    await execution.wait()

            assert caught.value.code is TransportErrorCode.UNAVAILABLE
            assert fatal.value.reason is BackendFatalReason.DEVICE_OOM
        finally:
            client.output_buffers.release(output)
            await close_stack(client, execution)

    run(scenario())
