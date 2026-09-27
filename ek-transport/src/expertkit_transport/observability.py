"""Optional OpenTelemetry providers; importing this module starts no exporter."""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from typing import Any, Literal

from expertkit_transport.tracing import TraceAttribute, TraceContext, TraceSpan


class SystemRandomIdGenerator:
    """Generate OTel IDs independently of deterministic model-process seeds."""

    def generate_span_id(self) -> int:
        """Return a nonzero 64-bit span ID."""
        return secrets.randbits(64) or 1

    def generate_trace_id(self) -> int:
        """Return a nonzero 128-bit trace ID."""
        return secrets.randbits(128) or 1

    def is_trace_id_random(self) -> bool:
        """Report uniform trace-ID bits to SDKs that expose the W3C random flag."""
        return True


class OpenTelemetryTracer:
    """Adapt a provider-owned tracer without installing a global provider."""

    def __init__(self, tracer: Any) -> None:
        from opentelemetry import context, trace
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

        self._tracer = tracer
        self._context = context
        self._trace = trace
        self._propagator = TraceContextTextMapPropagator()

    def current_span_is_recording(self) -> bool:
        """Return the current sampling decision."""
        return bool(self._trace.get_current_span().is_recording())

    def capture_context(self) -> TraceContext:
        """Capture the current parent for another execution context."""
        return self._context.get_current()

    def set_current_attributes(self, attributes: Mapping[str, TraceAttribute]) -> None:
        """Attach bounded fields without reading tensors."""
        self._trace.get_current_span().set_attributes(attributes)

    def inject_context(self) -> tuple[tuple[str, str], ...]:
        """Encode traceparent and tracestate, preserving unsampled parents."""
        carrier: dict[str, str] = {}
        self._propagator.inject(carrier)
        return tuple(carrier.items())

    def start_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> TraceSpan:
        """Start a detached span whose caller owns its end."""
        return self._tracer.start_span(name, context=context, attributes=attributes)

    def start_as_current_span(
        self,
        name: str,
        *,
        context: TraceContext | None = None,
        kind: Literal["internal", "client"] = "internal",
        attributes: Mapping[str, TraceAttribute] | None = None,
    ) -> Any:
        """Start a span and restore its parent when the scope exits."""
        return self._tracer.start_as_current_span(
            name, context=context, attributes=attributes, kind=self._trace.SpanKind[kind.upper()]
        )


class FrontendTracing:
    """Own optional sampled, asynchronous OTLP gRPC export for one process.

    An absent endpoint keeps optional packages unimported. Activate `tracer`
    with `use_tracer` around a benchmark or a model forward. Call `close` after
    all inference scopes finish, outside the computation hot path.
    """

    def __init__(
        self,
        endpoint: str | None,
        *,
        service_name: str = "expertkit-frontend",
        sample_ratio: float = 1.0,
    ) -> None:
        if not math.isfinite(sample_ratio) or not 0 <= sample_ratio <= 1:
            raise ValueError("trace sample ratio must be finite and between zero and one")
        self.tracer: OpenTelemetryTracer | None = None
        self._provider: Any = None
        if not endpoint:
            return
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "frontend tracing requires expertkit-transport[observability]"
            ) from error

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
            sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
            id_generator=SystemRandomIdGenerator(),
        )
        exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/"))
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                schedule_delay_millis=100,
                max_queue_size=16384,
                max_export_batch_size=512,
            )
        )
        self._provider = provider
        self.tracer = OpenTelemetryTracer(provider.get_tracer("expertkit-frontend"))

    def close(self) -> None:
        """Flush once at shutdown, outside model execution."""
        if self._provider is not None:
            self._provider.shutdown()
            self._provider = None
