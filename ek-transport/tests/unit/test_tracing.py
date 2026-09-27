"""Regression tests for scoped tracing across async and thread boundaries."""

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from expertkit_transport.observability import (
    FrontendTracing,
    OpenTelemetryTracer,
    SystemRandomIdGenerator,
)
from expertkit_transport.tracing import grpc_trace_metadata, trace_span, traced, use_tracer


def test_concurrent_batches_keep_their_parent_across_loop_threads() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OpenTelemetryTracer(provider.get_tracer("test"))

    @traced("layer")
    async def layer() -> None:
        await asyncio.sleep(0)
        with trace_span("rpc"):
            assert dict(grpc_trace_metadata())["traceparent"].endswith("-01")

    def caller(loop: asyncio.AbstractEventLoop) -> None:
        with use_tracer(tracer), trace_span("batch"):
            asyncio.run_coroutine_threadsafe(layer(), loop).result(timeout=3)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(2) as executor:
            await asyncio.gather(*(loop.run_in_executor(executor, caller, loop) for _ in range(2)))

    try:
        asyncio.run(scenario())
        spans = exporter.get_finished_spans()
        roots = [s for s in spans if s.name == "batch"]
        assert len(roots) == 2
        assert len({s.context.trace_id for s in roots}) == 2
        for root in roots:
            children = [s for s in spans if s.context.trace_id == root.context.trace_id]
            assert {s.name for s in children} == {"batch", "layer", "rpc"}
            by_name = {s.name: s for s in children}
            assert by_name["layer"].parent.span_id == root.context.span_id
            assert by_name["rpc"].parent.span_id == by_name["layer"].context.span_id
        assert grpc_trace_metadata() == ()
    finally:
        provider.shutdown()


def test_frontend_ids_are_unique_after_deterministic_process_seeding(monkeypatch) -> None:
    from opentelemetry.exporter.otlp.proto.grpc import trace_exporter

    exporters = []

    def exporter(*args, **kwargs):
        result = InMemorySpanExporter()
        exporters.append(result)
        return result

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", exporter)
    monkeypatch.setattr(random, "getrandbits", lambda bits: 1)
    runtimes = []
    try:
        for _ in range(2):
            runtime = FrontendTracing("http://unused:4317")
            runtimes.append(runtime)
            with use_tracer(runtime.tracer), trace_span("frontend.model_forward"):
                pass
        for runtime in runtimes:
            runtime.close()
        spans = [exporter.get_finished_spans()[0] for exporter in exporters]
        assert len({span.context.trace_id for span in spans}) == 2
        assert len({span.context.span_id for span in spans}) == 2
    finally:
        for runtime in runtimes:
            runtime.close()


def test_system_id_generator_declares_random_trace_ids() -> None:
    generator = SystemRandomIdGenerator()
    assert generator.is_trace_id_random()
    assert generator.generate_trace_id() != 0
    assert generator.generate_span_id() != 0


def test_unsampled_parent_is_propagated_and_context_restored_after_error() -> None:
    provider = TracerProvider(sampler=ALWAYS_OFF)
    tracer = OpenTelemetryTracer(provider.get_tracer("test"))
    try:
        with pytest.raises(ValueError, match="failure"), use_tracer(tracer), trace_span("batch"):
            assert dict(grpc_trace_metadata())["traceparent"].endswith("-00")
            raise ValueError("failure")
        assert grpc_trace_metadata() == ()
    finally:
        provider.shutdown()


def test_disabled_runtime_does_not_import_opentelemetry() -> None:
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from expertkit_transport.observability import FrontendTracing\n"
            "from expertkit_transport.tracing import trace_span, grpc_trace_metadata\n"
            "runtime = FrontendTracing(None)\n"
            "with trace_span('disabled'): pass\n"
            "assert grpc_trace_metadata() == ()\n"
            "runtime.close()\n"
            "assert not any(n.startswith('opentelemetry') for n in sys.modules)\n",
        ],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_sample_ratio_is_rejected(ratio: float) -> None:
    from expertkit_transport.observability import FrontendTracing

    with pytest.raises(ValueError, match="sample ratio"):
        FrontendTracing(None, sample_ratio=ratio)


def test_retry_stays_in_layer_trace_and_success_is_accumulated_once() -> None:
    import time
    from types import SimpleNamespace

    import torch

    from expertkit_transport.batches import RoutedLayerBatch
    from expertkit_transport.buffers import OutputPool
    from expertkit_transport.errors import TransportError, TransportErrorCode
    from expertkit_transport.routing import (
        RoundRobinSelector,
        TopologySnapshot,
        WorkerConnection,
        WorkerIdentity,
        execute_routed_layer,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OpenTelemetryTracer(provider.get_tracer("test"))
    calls = [0, 0]

    def worker(index):
        async def execute(batch, output, **kwargs):
            calls[index] += 1
            if index == 1 and calls[index] == 1:
                raise TransportError(TransportErrorCode.BUSY, retryable=True)
            output.fill_(index + 1)

        return WorkerConnection(
            WorkerIdentity(str(index), "start"),
            SimpleNamespace(execute=execute),
            2,
            1,
            1,
        )

    async def scenario():
        workers = [worker(i) for i in range(2)]
        snapshot = TopologySnapshot(7, 1, {(0, i): (w,) for i, w in enumerate(workers)})

        async def refresh(*args, **kwargs):
            return snapshot

        topology = SimpleNamespace(current=lambda instance_id: snapshot, refresh=refresh)
        pools = {
            w.identity: OutputPool(
                max_batch_tokens=2,
                hidden_dim=2,
                dtype=torch.float32,
                device="cpu",
                capacity=2,
            )
            for w in workers
        }
        batch = RoutedLayerBatch(
            7,
            0,
            torch.ones(1, 2),
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.full((1, 2), 0.5),
            (0, 1),
        )
        try:
            with use_tracer(tracer), trace_span("layer"):
                result = await execute_routed_layer(
                    batch,
                    topology,
                    RoundRobinSelector(),
                    pools,
                    monotonic_deadline=time.monotonic() + 3,
                )
                torch.testing.assert_close(result, torch.full((1, 2), 3.0))
        finally:
            for pool in pools.values():
                await pool.close()

    try:
        asyncio.run(scenario())
        assert calls == [1, 2]
        spans = exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "layer")
        assert all(s.context.trace_id == root.context.trace_id for s in spans)
        retry = next(s for s in spans if s.name == "transport.retry")
        assert retry.attributes["expertkit.attempt"] == 2
        assert retry.parent.span_id == root.context.span_id
        attempt = next(s for s in spans if s.name == "transport.attempt")
        assert attempt.attributes["expertkit.attempt"] == 1
        assert attempt.parent.span_id == root.context.span_id
    finally:
        provider.shutdown()


def test_cancelled_operation_ends_span_and_restores_parent() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OpenTelemetryTracer(provider.get_tracer("test"))

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        @traced("cancelled_rpc")
        async def call():
            started.set()
            await release.wait()

        with use_tracer(tracer), trace_span("batch"):
            parent = grpc_trace_metadata()
            task = asyncio.create_task(call())
            async with asyncio.timeout(2):
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert grpc_trace_metadata() == parent
        assert grpc_trace_metadata() == ()

    try:
        asyncio.run(scenario())
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["cancelled_rpc", "batch"]
        assert spans[0].parent.span_id == spans[1].context.span_id
        assert spans[0].end_time <= spans[1].end_time
    finally:
        provider.shutdown()
