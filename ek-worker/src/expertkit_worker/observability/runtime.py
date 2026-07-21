"""Optional Prometheus listener and asynchronous OpenTelemetry export."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Protocol

from expertkit_transport.tracing import TraceAttribute, TraceContext, Tracer, TraceSpan

from expertkit_worker.config.models import ObservabilityConfig
from expertkit_worker.observability.api import NoopWorkerMetrics, WorkerMetrics


class WorkerObservability(Protocol):
    """Own optional observability resources for one Worker process."""

    @property
    def metrics(self) -> WorkerMetrics:
        """Return the dependency-free metrics recorder."""

    @property
    def grpc_interceptors(self) -> Sequence[Any]:
        """Return gRPC server interceptors installed before server startup."""

    @property
    def tracer(self) -> Tracer | None:
        """Return optional tracing operations for Transport and execution."""

    async def start(self) -> None:
        """Start optional listeners."""

    async def close(self) -> None:
        """Flush exporters and stop optional listeners."""


class _NoopObservability:
    def __init__(self) -> None:
        self._metrics = NoopWorkerMetrics()

    @property
    def metrics(self) -> WorkerMetrics:
        return self._metrics

    @property
    def grpc_interceptors(self) -> Sequence[Any]:
        return ()

    @property
    def tracer(self) -> Tracer | None:
        return None

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


class _PrometheusMetrics:
    """Hold a private low-cardinality Prometheus registry for one process."""

    def __init__(self, registry: Any) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        self._batches = Counter(
            "expertkit_worker_batches_total",
            "Worker batches that entered active execution.",
            ("outcome",),
            registry=registry,
        )
        self._batch_duration = Histogram(
            "expertkit_worker_batch_duration_seconds",
            "Host-observed Worker batch duration without device-only timing.",
            registry=registry,
        )
        self._active = Gauge(
            "expertkit_worker_active_batches",
            "Worker batches currently in active execution.",
            registry=registry,
        )
        self._rejections = Counter(
            "expertkit_worker_batch_rejections_total",
            "Worker batch rejections by bounded error reason.",
            ("reason",),
            registry=registry,
        )
        self._pending = Gauge(
            "expertkit_worker_pending_batches",
            "Decoded Worker batches retained in the Transport waiting area.",
            registry=registry,
        )
        self._weight_sources = Counter(
            "expertkit_worker_weight_source_attempts_total",
            "Weight-source lookup attempts by source and outcome.",
            ("source", "outcome"),
            registry=registry,
        )
        self._expert_states = Counter(
            "expertkit_worker_expert_state_changes_total",
            "Reportable expert-state transitions.",
            ("state",),
            registry=registry,
        )
        self._device_weight_bytes = Gauge(
            "expertkit_worker_device_weight_bytes",
            "Accounted bytes held by ready device weights.",
            registry=registry,
        )

    def batch_started(self) -> None:
        self._active.inc()

    def batch_finished(self, *, fatal: bool, duration_seconds: float) -> None:
        self._active.dec()
        self._batches.labels(outcome="fatal" if fatal else "completed").inc()
        self._batch_duration.observe(max(0.0, duration_seconds))

    def batch_rejected(self, reason: str) -> None:
        self._rejections.labels(reason=reason).inc()

    def pending_batches_changed(self, count: int) -> None:
        self._pending.set(count)

    def weight_source_result(self, source: str, *, success: bool) -> None:
        self._weight_sources.labels(
            source=source,
            outcome="success" if success else "failure",
        ).inc()

    def expert_state_changed(self, state: str) -> None:
        self._expert_states.labels(state=state).inc()

    def device_weight_bytes_changed(self, byte_count: int) -> None:
        self._device_weight_bytes.set(byte_count)


class _OpenTelemetryTracer:
    """Adapt one configured OpenTelemetry tracer to the shared tracing boundary."""

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
    ) -> Any:
        return self._tracer.start_as_current_span(
            name,
            context=context,
            attributes=None if attributes is None else dict(attributes),
        )


class _OptionalObservability:
    def __init__(self, config: ObservabilityConfig, *, worker_id: str) -> None:
        self._listen = config.prometheus.listen if config.prometheus.enabled else None
        self._metrics_server: Any = None
        self._metrics_thread: Any = None
        self._started = False
        self._closed = False

        if config.prometheus.enabled:
            from prometheus_client import CollectorRegistry

            self._registry: Any = CollectorRegistry()
            self._metrics: WorkerMetrics = _PrometheusMetrics(self._registry)
        else:
            self._registry = None
            self._metrics = NoopWorkerMetrics()

        self._tracer_provider: Any = None
        self._tracer: Tracer | None = None
        self._interceptors: tuple[Any, ...] = ()
        if config.tracing.enabled:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.instrumentation.grpc import aio_server_interceptor
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

            endpoint = str(config.tracing.endpoint).rstrip("/")
            provider = TracerProvider(
                resource=Resource.create(
                    {
                        "service.name": "expertkit-worker",
                        "service.instance.id": worker_id,
                    }
                ),
                sampler=ParentBased(TraceIdRatioBased(config.tracing.sample_ratio)),
            )
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    schedule_delay_millis=100,
                    max_queue_size=16384,
                    max_export_batch_size=512,
                )
            )
            self._tracer_provider = provider
            self._tracer = _OpenTelemetryTracer(
                provider.get_tracer("expertkit-worker"),
                otel_context,
                otel_trace,
            )
            self._interceptors = (aio_server_interceptor(tracer_provider=provider),)

    @property
    def metrics(self) -> WorkerMetrics:
        return self._metrics

    @property
    def grpc_interceptors(self) -> Sequence[Any]:
        return self._interceptors

    @property
    def tracer(self) -> Tracer | None:
        return self._tracer

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Worker observability is closed")
        if self._started:
            return
        self._started = True
        if self._listen is None:
            return
        from prometheus_client import start_http_server

        host, port = _split_address(self._listen)
        self._metrics_server, self._metrics_thread = start_http_server(
            port,
            addr=host,
            registry=self._registry,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        if self._metrics_server is not None and self._metrics_thread is not None:
            await loop.run_in_executor(
                None,
                partial(
                    _stop_metrics_server,
                    self._metrics_server,
                    self._metrics_thread,
                ),
            )
        if self._tracer_provider is not None:
            await loop.run_in_executor(None, self._tracer_provider.shutdown)


def create_observability(
    config: ObservabilityConfig,
    *,
    worker_id: str,
) -> WorkerObservability:
    """Build disabled state or import the optional exporter stack on demand.

    Raises:
        RuntimeError: An enabled integration is missing its optional dependency.
    """

    if not config.prometheus.enabled and not config.tracing.enabled:
        return _NoopObservability()
    try:
        return _OptionalObservability(config, worker_id=worker_id)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "enabled Worker observability requires the locked observability extra"
        ) from error


def _split_address(value: str) -> tuple[str, int]:
    if value.startswith("["):
        closing = value.index("]")
        return value[1:closing], int(value[closing + 2 :])
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _stop_metrics_server(server: Any, thread: Any) -> None:
    server.shutdown()
    server.server_close()
    thread.join()
