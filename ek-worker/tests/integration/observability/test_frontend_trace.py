"""Trace real Torch frontend, gRPC fanout and expert execution as one batch."""

import asyncio

import pytest
import torch

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("opentelemetry.instrumentation.grpc")

from expertkit_torch.client import RoutedMoEClient
from expertkit_transport import client as transport_client
from expertkit_transport.buffers import OutputPool
from expertkit_transport.controller.instance import ResolvedDefaultInstance
from expertkit_transport.observability import OpenTelemetryTracer
from expertkit_transport.routing import TopologySnapshot, WorkerConnection, WorkerIdentity
from expertkit_transport.tracing import trace_span, use_tracer
from expertkit_transport.transports.base import BatchBufferConfig, WorkerEndpointConfig
from expertkit_transport.transports.grpc import GrpcWorkerBatchReceiver, GrpcWorkerTransport
from opentelemetry.instrumentation.grpc import aio_server_interceptor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights
from expertkit_worker.execution import WorkerExecutor
from expertkit_worker.factory import _create_device_wiring
from expertkit_worker.weights import ReadyWeightTable


@pytest.mark.parametrize("sample_ratio", [1.0, 0.0])
def test_model_batch_connects_all_layers_and_workers(
    monkeypatch,
    sample_ratio: float,
) -> None:
    worker_count = 2
    exporter = InMemorySpanExporter()
    providers = []

    def provider(service, ratio=1.0):
        result = TracerProvider(
            resource=Resource.create({"service.name": service}),
            sampler=ParentBased(TraceIdRatioBased(ratio)),
        )
        result.add_span_processor(SimpleSpanProcessor(exporter))
        providers.append(result)
        return result

    frontend_provider = provider("expertkit-frontend", sample_ratio)
    frontend_tracer = OpenTelemetryTracer(frontend_provider.get_tracer("frontend"))
    spec = WorkerEndpointConfig(7, 2, 3, 2, 2, 3, torch.float32)

    async def scenario():
        executions = []
        endpoints = []
        for _ in range(worker_count):
            worker_provider = provider("expertkit-worker")
            tracer = OpenTelemetryTracer(worker_provider.get_tracer("worker"))
            receiver = GrpcWorkerBatchReceiver(
                "127.0.0.1:0",
                spec,
                max_active_batches=1,
                max_pending_batches=4,
                interceptors=(aio_server_interceptor(tracer_provider=worker_provider),),
                tracer=tracer,
            )
            weights = ReadyWeightTable(2, 3)
            for layer in range(2):
                for expert in range(3):
                    weights.publish(
                        layer,
                        expert,
                        TorchExpertWeights(
                            torch.eye(2),
                            torch.eye(2),
                            torch.eye(2),
                        ),
                    )
            wiring = _create_device_wiring("cpu")
            backend = TorchBackend(
                hidden_dim=2,
                intermediate_dim=2,
                top_k=3,
                dtype=torch.float32,
                runtime=wiring.runtime,
                acquire_many=weights.acquire_many,
            )
            executor = WorkerExecutor(
                receiver,
                backend,
                create_slot=wiring.create_slot,
                instance_id=7,
                buffer_config=BatchBufferConfig(2, 2, 3, torch.float32, torch.device("cpu")),
                slot_count=1,
                tracer=tracer,
            )
            executions.append(executor)
            await executor.start()
            endpoints.append(f"127.0.0.1:{receiver.bound_port}")

        class StaticTopology:
            """Replace only the external Controller with pre-published routes."""

            def __init__(self, *args, **kwargs):
                self.targets = tuple(
                    WorkerConnection(
                        WorkerIdentity(f"worker-{i}", f"start-{i}"),
                        GrpcWorkerTransport(endpoint, spec, max_in_flight=2, device="cpu"),
                        2,
                        1,
                        1,
                    )
                    for i, endpoint in enumerate(endpoints)
                )
                self.pools = {
                    t.identity: OutputPool(
                        max_batch_tokens=2,
                        hidden_dim=2,
                        dtype=torch.float32,
                        device="cpu",
                        capacity=2,
                    )
                    for t in self.targets
                }
                self.snapshot = TopologySnapshot(
                    7,
                    1,
                    {
                        (layer, expert): (self.targets[expert % worker_count],)
                        for layer in range(2)
                        for expert in range(3)
                    },
                )

            async def start(self, **kwargs):
                for target in self.targets:
                    await target.transport.start()

            def current(self, instance_id):
                return self.snapshot

            async def close(self):
                for target in self.targets:
                    await target.transport.close()
                for pool in self.pools.values():
                    await pool.close()

        async def resolve(*args, **kwargs):
            return ResolvedDefaultInstance(7, "synthetic", "test")

        monkeypatch.setattr(transport_client, "resolve_default_instance", resolve)
        monkeypatch.setattr(transport_client, "ControllerTopologyWatcher", StaticTopology)
        client = RoutedMoEClient(
            "controller",
            num_layers=2,
            experts_per_layer=3,
            hidden_dim=2,
            top_k=3,
        )

        def forward():
            hidden = torch.tensor([[0.5, -0.25], [0.75, 0.125]])
            with use_tracer(frontend_tracer), trace_span("frontend.model_forward"):
                for layer in range(2):
                    result = client.forward_layer(
                        layer_id=layer,
                        hidden_states=hidden,
                        expert_ids=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32),
                        routing_weights=torch.full((2, 3), 1 / 3),
                    )
                    torch.testing.assert_close(result, torch.nn.functional.silu(hidden) * hidden)

        try:
            async with asyncio.timeout(15):
                await asyncio.gather(asyncio.to_thread(forward), asyncio.to_thread(forward))
        finally:
            await asyncio.to_thread(client.close)
            for executor in executions:
                await executor.close()

    try:
        asyncio.run(scenario())
        spans = exporter.get_finished_spans()
        if sample_ratio == 0:
            assert not spans
            return
        roots = [s for s in spans if s.parent is None]
        assert len(roots) == 2
        assert {s.name for s in roots} == {"frontend.model_forward"}
        for root in roots:
            children = [s for s in spans if s.context.trace_id == root.context.trace_id]
            by_id = {s.context.span_id: s for s in children}
            assert all(s.parent is None or s.parent.span_id in by_id for s in children)
            assert {s.resource.attributes["service.name"] for s in children} == {
                "expertkit-frontend",
                "expertkit-worker",
            }
            layers = [s for s in children if s.name == "transport.layer"]
            assert len(layers) == 2
            assert any(s.name == "worker.backend.submit" for s in children)
            rpcs = [s for s in children if s.name == "transport.grpc.rpc"]
            assert len(rpcs) == 2 * worker_count
            for rpc in rpcs:
                assert rpc.kind.name == "CLIENT"
                server = [
                    s for s in children if s.parent and s.parent.span_id == rpc.context.span_id
                ]
                assert len(server) == 1
                assert server[0].name == "/ek.worker.v2.ComputationService/Execute"
    finally:
        for item in providers:
            item.shutdown()
