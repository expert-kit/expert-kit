"""Pinned forward-context instrumentation, independent of device availability."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from expertkit_transport.observability import OpenTelemetryTracer
from expertkit_transport.tracing import trace_span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from expertkit_vllm import tracing


def test_forward_wrapper_keeps_full_batch_and_restores_original_context(monkeypatch):
    module = ModuleType("vllm.forward_context")
    current = []

    @contextmanager
    def original(context):
        current.append(context)
        try:
            yield
        finally:
            current.pop()

    module.override_forward_context = original
    monkeypatch.setitem(sys.modules, "vllm.forward_context", module)
    package = ModuleType("vllm")
    package.forward_context = module
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setenv("EK_TRACE_ENDPOINT", "http://unused:4317")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OpenTelemetryTracer(provider.get_tracer("test"))
    calls = []
    close_hooks = []

    def runtime(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(tracer=otel, close=lambda: None)

    monkeypatch.setattr(tracing, "FrontendTracing", runtime)
    monkeypatch.setattr(tracing.atexit, "register", close_hooks.append)
    try:
        tracing.install_forward_tracing()
        wrapper = module.override_forward_context
        tracing.install_forward_tracing()
        assert module.override_forward_context is wrapper
        assert not calls
        context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=32))
        with pytest.raises(ValueError, match="model failure"), wrapper(context):
            assert current == [context]
            for _ in range(2):
                with trace_span("frontend.moe"):
                    pass
            raise ValueError("model failure")
        assert not current
        with wrapper(forward_context=context):
            pass
        assert len(calls) == len(close_hooks) == 1
        spans = exporter.get_finished_spans()
        roots = [s for s in spans if s.name == "frontend.model_forward"]
        assert len(roots) == 2
        root = roots[0]
        assert root.attributes["expertkit.token_count"] == 32
        assert root.status.status_code.name == "ERROR"
        assert all(
            s.parent.span_id == root.context.span_id for s in spans if s.name == "frontend.moe"
        )
        assert all(s.parent is None for s in roots)
    finally:
        provider.shutdown()


def test_disabled_forward_tracing_does_not_import_vllm(monkeypatch):
    monkeypatch.delenv("EK_TRACE_ENDPOINT", raising=False)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", None)
    tracing.install_forward_tracing()


def test_pinned_vllm_forward_context_preserves_real_context(monkeypatch):
    fc = pytest.importorskip("vllm.forward_context")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    runtime = SimpleNamespace(
        tracer=OpenTelemetryTracer(provider.get_tracer("test")),
        close=lambda: None,
    )
    monkeypatch.setenv("EK_TRACE_ENDPOINT", "http://unused:4317")
    monkeypatch.setattr(tracing, "FrontendTracing", lambda *a, **k: runtime)
    monkeypatch.setattr(tracing.atexit, "register", lambda hook: None)
    original = fc.override_forward_context
    context = fc.ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        batch_descriptor=fc.BatchDescriptor(num_tokens=4),
    )
    try:
        tracing.install_forward_tracing()
        with fc.override_forward_context(forward_context=context):
            assert fc.get_forward_context() is context
            with trace_span("frontend.moe"):
                pass
        root = next(s for s in exporter.get_finished_spans() if s.parent is None)
        assert root.name == "frontend.model_forward"
        assert root.attributes["expertkit.token_count"] == 4
    finally:
        fc.override_forward_context = original
        provider.shutdown()
