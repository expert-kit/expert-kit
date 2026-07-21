"""Smoke tests for optional metrics and asynchronous trace export."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import grpc
import pytest
import torch
from aiohttp import ClientSession
from expertkit_transport.batches import WorkerBatch
from expertkit_transport.transports import WorkerEndpointConfig
from expertkit_transport.transports.base import BatchBufferConfig
from expertkit_transport.transports.grpc import (
    GrpcWorkerBatchReceiver,
    decode_response,
    encode_request,
)

from expertkit_worker.backends import (
    BackendBatch,
    BackendCapabilities,
    BackendCompletion,
    BackendResourceEstimate,
    CompletedSubmission,
    ComputeBackend,
)
from expertkit_worker.config.models import ObservabilityConfig
from expertkit_worker.execution import WorkerExecutor
from expertkit_worker.observability import create_observability

pytest.importorskip("prometheus_client")
pytest.importorskip("opentelemetry.sdk")

from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)

_EXECUTE_METHOD = "/ek.worker.v2.ComputationService/Execute"
_TRACE_ID = "0123456789abcdef0123456789abcdef"
_REMOTE_PARENT_SPAN_ID = "0123456789abcdef"


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _spec() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(
        instance_id=7,
        num_layers=1,
        experts_per_layer=1,
        max_batch_tokens=2,
        hidden_dim=2,
        top_k=1,
        dtype=torch.float32,
    )


def _batch() -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=0,
        topology_version=1,
        hidden_states=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        token_indices=None,
        expert_ids=torch.tensor([[0]], dtype=torch.int32),
        routing_weights=torch.ones((1, 1), dtype=torch.float32),
        distinct_expert_ids=(0,),
    )


class _DoubleBackend(ComputeBackend):
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
        torch.mul(batch.hidden_states, 2, out=prepared_output)
        return CompletedSubmission()


class _TraceCollector(trace_service_pb2_grpc.TraceServiceServicer):
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.received = asyncio.Event()
        self.release = asyncio.Event()

    async def Export(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> trace_service_pb2.ExportTraceServiceResponse:
        self.requests.append(request)
        self.received.set()
        await self.release.wait()
        return trace_service_pb2.ExportTraceServiceResponse()


def test_prometheus_listener_exposes_low_cardinality_worker_metrics() -> None:
    async def scenario() -> None:
        port = _unused_port()
        config = ObservabilityConfig.model_validate(
            {
                "prometheus": {
                    "enabled": True,
                    "listen": f"127.0.0.1:{port}",
                }
            }
        )
        observability = create_observability(config, worker_id="worker-0")
        assert observability.tracer is None
        await observability.start()
        try:
            metrics = observability.metrics
            metrics.batch_started()
            metrics.batch_finished(fatal=False, duration_seconds=0.01)
            metrics.batch_rejected("busy")
            metrics.pending_batches_changed(2)
            metrics.weight_source_result("disk", success=False)
            metrics.expert_state_changed("ready")
            metrics.device_weight_bytes_changed(4096)

            async with (
                ClientSession() as session,
                session.get(f"http://127.0.0.1:{port}/metrics") as response,
            ):
                assert response.status == 200
                body = await response.text()

            assert 'expertkit_worker_batches_total{outcome="completed"} 1.0' in body
            assert 'expertkit_worker_batch_rejections_total{reason="busy"} 1.0' in body
            assert "expertkit_worker_pending_batches 2.0" in body
            assert (
                'expertkit_worker_weight_source_attempts_total{outcome="failure",source="disk"} 1.0'
            ) in body
            assert "expertkit_worker_device_weight_bytes 4096.0" in body
        finally:
            await observability.close()

    asyncio.run(scenario())


def test_tracing_extracts_parent_context_and_exports_off_the_rpc_path() -> None:
    async def scenario() -> None:
        collector = _TraceCollector()
        collector_server = grpc.aio.server()
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(
            collector,
            collector_server,
        )
        collector_port = collector_server.add_insecure_port("127.0.0.1:0")
        await collector_server.start()

        config = ObservabilityConfig.model_validate(
            {
                "tracing": {
                    "enabled": True,
                    "endpoint": f"http://127.0.0.1:{collector_port}",
                    "sample_ratio": 1,
                }
            }
        )
        observability = create_observability(config, worker_id="worker-0")
        server = GrpcWorkerBatchReceiver(
            "127.0.0.1:0",
            _spec(),
            max_active_batches=1,
            max_pending_batches=1,
            interceptors=observability.grpc_interceptors,
            tracer=observability.tracer,
        )
        execution = WorkerExecutor(
            server,
            _DoubleBackend(),
            instance_id=7,
            buffer_config=BatchBufferConfig(2, 2, 1, torch.float32, "cpu"),
            slot_count=1,
            tracer=observability.tracer,
        )
        channel: grpc.aio.Channel | None = None
        await observability.start()
        await execution.start()
        try:
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}")
            execute = channel.unary_unary(
                _EXECUTE_METHOD,
                request_serializer=lambda payload: payload,
                response_deserializer=lambda payload: payload,
            )
            payload = await execute(
                encode_request(_batch(), _spec()),
                metadata=(("traceparent", f"00-{_TRACE_ID}-{_REMOTE_PARENT_SPAN_ID}-01"),),
            )
            torch.testing.assert_close(
                decode_response(payload, 1, _spec()),
                torch.tensor([[2.0, 4.0]], dtype=torch.float32),
            )

            async with asyncio.timeout(2):
                await collector.received.wait()
            assert not collector.release.is_set()
            collector.release.set()
        finally:
            collector.release.set()
            if channel is not None:
                await channel.close()
            await execution.close()
            await observability.close()
            await collector_server.stop(None)

        spans = [
            span
            for request in collector.requests
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        ]
        traced = [span for span in spans if span.trace_id == bytes.fromhex(_TRACE_ID)]
        assert traced, [
            (span.name, span.trace_id.hex(), span.parent_span_id.hex()) for span in spans
        ]
        by_name = {span.name: span for span in traced}
        expected = {
            "worker.request.decode",
            "worker.request.wait",
            "worker.batch.execute",
            "worker.input.prepare",
            "worker.backend.submit",
            "worker.backend.wait",
            "worker.output.prepare",
            "worker.response.encode",
        }
        assert expected <= by_name.keys()

        server_span = next(
            span for span in traced if span.parent_span_id == bytes.fromhex(_REMOTE_PARENT_SPAN_ID)
        )
        for name in ("worker.request.decode", "worker.request.wait", "worker.batch.execute"):
            assert by_name[name].parent_span_id == server_span.span_id
        for name in (
            "worker.input.prepare",
            "worker.backend.submit",
            "worker.backend.wait",
            "worker.output.prepare",
            "worker.response.encode",
        ):
            assert by_name[name].parent_span_id == by_name["worker.batch.execute"].span_id

    asyncio.run(scenario())
