"""Dependency-free tracing boundary shared by Transport and Worker execution."""

from __future__ import annotations

import atexit
import os
import threading
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

type TraceAttribute = str | bool | int | float
type TraceContext = object


class TraceSpan(Protocol):
    """Expose the span operations used on the computation path."""

    def is_recording(self) -> bool:
        """Return whether this sampled span records attributes and events."""

    def set_attribute(self, key: str, value: TraceAttribute) -> None:
        """Attach one bounded scalar value to the span."""

    def end(self) -> None:
        """Finish a detached span exactly once."""


class Tracer(Protocol):
    """Create spans without exposing an OpenTelemetry package dependency."""

    def current_span_is_recording(self) -> bool:
        """Return whether the current request span was sampled."""

    def capture_context(self) -> TraceContext:
        """Capture the current parent context for another task or thread."""

    def start_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> TraceSpan:
        """Start a detached span that the caller will end."""

    def start_as_current_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        """Start a span and make it current for the context-manager lifetime."""


class _OpenTelemetryTracer:
    """Adapt an OpenTelemetry tracer to the local dependency-free Protocol."""

    def __init__(self, tracer: Any, context_api: Any, trace_api: Any) -> None:
        self._tracer = tracer
        self._context_api = context_api
        self._trace_api = trace_api

    def current_span_is_recording(self) -> bool:
        return bool(self._trace_api.get_current_span().is_recording())

    def capture_context(self) -> TraceContext:
        return self._context_api.get_current()

    def start_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> TraceSpan:
        return self._tracer.start_span(
            name,
            context=context,
            attributes=None if attributes is None else dict(attributes),
        )

    def start_as_current_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> AbstractContextManager[TraceSpan]:
        return self._tracer.start_as_current_span(
            name,
            context=context,
            attributes=None if attributes is None else dict(attributes),
        )


_TRACING_LOCK = threading.Lock()
_TRACER_PROVIDER: Any | None = None
_TRACER: Tracer | None = None
_TRACING_SHUT_DOWN = False


def initialize_frontend_tracing() -> Tracer | None:
    """Initialize optional Frontend OpenTelemetry export from OTEL env vars."""

    global _TRACER_PROVIDER, _TRACER, _TRACING_SHUT_DOWN
    with _TRACING_LOCK:
        if _TRACER is not None:
            return _TRACER
        if _TRACING_SHUT_DOWN:
            return None
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not endpoint:
            return None

        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ALWAYS_ON
        except ImportError:
            return None

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "frontend"),
                    "expertkit.component": "frontend",
                }
            ),
            sampler=ALWAYS_ON,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint.rstrip("/"), insecure=True),
                schedule_delay_millis=100,
                max_queue_size=16384,
                max_export_batch_size=512,
            )
        )
        _TRACER_PROVIDER = provider
        _TRACER = _OpenTelemetryTracer(
            provider.get_tracer("expertkit-transport"),
            otel_context,
            otel_trace,
        )
        return _TRACER


def get_frontend_tracer() -> Tracer | None:
    """Return the optional Frontend tracer, initializing it lazily."""

    tracer = _TRACER
    return tracer if tracer is not None else initialize_frontend_tracing()


def trace_span(
    tracer: Tracer | None,
    name: str,
    *,
    context: TraceContext | None = None,
    attributes: Mapping[str, TraceAttribute] | None = None,
) -> AbstractContextManager[TraceSpan | None]:
    """Start a span when tracing is enabled, otherwise return a no-op context."""

    if tracer is None:
        return nullcontext(None)
    return tracer.start_as_current_span(name, context=context, attributes=attributes)


def frontend_request_span(
    *,
    attributes: Mapping[str, TraceAttribute] | None = None,
) -> AbstractContextManager[TraceSpan | None]:
    """Create the root span for one caller-defined inference request."""

    return trace_span(get_frontend_tracer(), "frontend.request", attributes=attributes)


def current_trace_metadata() -> tuple[tuple[str, str], ...]:
    """Inject only W3C trace context into gRPC metadata."""

    try:
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    except ImportError:
        return ()
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    traceparent = carrier.get("traceparent")
    return () if not traceparent else (("traceparent", traceparent),)


def extract_trace_context(metadata: object) -> TraceContext | None:
    """Extract W3C trace context from gRPC invocation metadata."""

    try:
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    except ImportError:
        return None

    carrier: dict[str, str] = {}
    for entry in metadata or ():
        key = getattr(entry, "key", None)
        value = getattr(entry, "value", None)
        if key is None or value is None:
            try:
                key, value = entry
            except (TypeError, ValueError):
                continue
        if isinstance(key, str) and isinstance(value, str):
            carrier[key.lower()] = value
    if "traceparent" not in carrier:
        return None
    return TraceContextTextMapPropagator().extract(carrier)


def incoming_trace_context(tracer: Tracer | None, metadata: object) -> TraceContext | None:
    """Capture an instrumented server span or extract the remote W3C parent."""

    if tracer is None:
        return None
    if tracer.current_span_is_recording():
        return tracer.capture_context()
    return extract_trace_context(metadata)


def flush_tracing() -> None:
    """Force-export buffered Frontend spans without closing the provider."""

    provider = _TRACER_PROVIDER
    if provider is not None:
        provider.force_flush()


def shutdown_tracing() -> None:
    """Flush and close Frontend tracing exactly once during process shutdown."""

    global _TRACER_PROVIDER, _TRACER, _TRACING_SHUT_DOWN
    with _TRACING_LOCK:
        if _TRACING_SHUT_DOWN:
            return
        _TRACING_SHUT_DOWN = True
        provider = _TRACER_PROVIDER
        _TRACER_PROVIDER = None
        _TRACER = None
    if provider is not None:
        provider.shutdown()


atexit.register(shutdown_tracing)
