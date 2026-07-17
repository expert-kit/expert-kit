"""End-to-end routed-layer tests across two real gRPC Workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch
from expertkit_transport.adapters.grpc import (
    GrpcBatchSpec,
    GrpcWorkerServer,
    GrpcWorkerTransport,
)
from expertkit_transport.buffers import OutputPool
from expertkit_transport.contracts import OutputSpec, RoutedLayerBatch, WorkerPositionSpec
from expertkit_transport.orchestration import (
    RoundRobinSelector,
    TopologySnapshot,
    WorkerIdentity,
    WorkerTarget,
    execute_routed_layer,
)

from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights
from expertkit_worker.execution import WorkerExecution
from expertkit_worker.weights import ReadyWeightTable

_INSTANCE_ID = 7
_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 6
_TOP_K = 3
_MAX_BATCH_TOKENS = 4


@dataclass(slots=True)
class _RunningWorker:
    identity: WorkerIdentity
    server: GrpcWorkerServer
    transport: GrpcWorkerTransport
    execution: WorkerExecution
    ready: ReadyWeightTable[TorchExpertWeights]

    @property
    def target(self) -> WorkerTarget:
        """Return the Frontend routing target for this process start."""

        return WorkerTarget(
            identity=self.identity,
            transport=self.transport,
            max_batch_tokens=_MAX_BATCH_TOKENS,
            max_active_batches=1,
            max_pending_batches=1,
        )

    async def close(self) -> None:
        """Close the client before stopping its Worker receiver."""

        try:
            await self.transport.close()
        finally:
            await self.execution.close()


class _StaticTopology:
    def __init__(self, snapshot: TopologySnapshot) -> None:
        self._snapshot = snapshot

    def current(self, instance_id: int) -> TopologySnapshot:
        if instance_id != self._snapshot.instance_id:
            raise AssertionError("test requested a different model instance")
        return self._snapshot

    async def refresh(
        self,
        instance_id: int,
        *,
        observed_version: int,
        monotonic_deadline: float,
    ) -> TopologySnapshot:
        raise AssertionError("the healthy two-Worker path must not refresh Topology")


def _weight(seed: int) -> TorchExpertWeights:
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return (torch.randn(rows, columns, generator=generator) / 5).contiguous()

    return TorchExpertWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def _batch_spec() -> GrpcBatchSpec:
    return GrpcBatchSpec(
        instance_id=_INSTANCE_ID,
        num_layers=1,
        experts_per_layer=3,
        max_batch_tokens=_MAX_BATCH_TOKENS,
        hidden_dim=_HIDDEN_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
    )


async def _start_worker(
    worker_id: str,
    expert_id: int,
    weight: TorchExpertWeights,
) -> _RunningWorker:
    ready: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 3)
    ready.publish(0, expert_id, weight)
    server = GrpcWorkerServer(
        "127.0.0.1:0",
        _batch_spec(),
        max_active_batches=1,
        max_pending_batches=1,
    )
    backend = TorchBackend(
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        top_k=_TOP_K,
        dtype=torch.float32,
        device="cpu",
        acquire_many=ready.acquire_many,
    )
    execution = WorkerExecution(
        server,
        backend,
        instance_id=_INSTANCE_ID,
        position_spec=WorkerPositionSpec(
            max_batch_tokens=_MAX_BATCH_TOKENS,
            hidden_dim=_HIDDEN_DIM,
            top_k=_TOP_K,
            dtype=torch.float32,
            device="cpu",
        ),
        active_positions=1,
    )
    await server.start()
    await execution.start()
    transport = GrpcWorkerTransport(
        f"127.0.0.1:{server.bound_port}",
        _batch_spec(),
        max_in_flight=2,
    )
    await transport.start()
    return _RunningWorker(
        identity=WorkerIdentity(worker_id, f"{worker_id}-start"),
        server=server,
        transport=transport,
        execution=execution,
        ready=ready,
    )


def _reference(
    batch: RoutedLayerBatch,
    weights: dict[int, TorchExpertWeights],
) -> torch.Tensor:
    result = torch.zeros_like(batch.hidden_states, dtype=torch.float32)
    for token_index in range(batch.token_count):
        token = batch.hidden_states[token_index]
        for route_index in range(batch.top_k):
            expert_id = int(batch.expert_ids[token_index, route_index])
            if expert_id < 0:
                continue
            weight = weights[expert_id]
            hidden = torch.nn.functional.silu(token @ weight.gate_proj.T)
            hidden.mul_(token @ weight.up_proj.T)
            output = hidden @ weight.down_proj.T
            result[token_index].add_(output * batch.routing_weights[token_index, route_index])
    return result


def test_two_workers_execute_and_aggregate_one_routed_layer() -> None:
    async def scenario() -> None:
        weights = {0: _weight(11), 1: _weight(29)}
        worker_a = await _start_worker("worker-a", 0, weights[0])
        worker_b = await _start_worker("worker-b", 1, weights[1])
        workers = (worker_a, worker_b)
        pools: dict[WorkerIdentity, OutputPool] = {}
        try:
            targets = tuple(worker.target for worker in workers)
            pools = {
                target.identity: OutputPool(
                    target.transport.output_buffers,
                    OutputSpec(
                        _MAX_BATCH_TOKENS,
                        _HIDDEN_DIM,
                        torch.float32,
                        "cpu",
                    ),
                    capacity=target.max_in_flight,
                )
                for target in targets
            }
            topology = TopologySnapshot(
                instance_id=_INSTANCE_ID,
                version=4,
                routes={(0, 0): (targets[0],), (0, 1): (targets[1],)},
            )
            batch = RoutedLayerBatch(
                instance_id=_INSTANCE_ID,
                layer_id=0,
                hidden_states=torch.tensor(
                    [
                        [0.2, -0.4, 0.8, 0.5],
                        [-0.1, 0.7, 0.3, -0.6],
                        [0.9, 0.1, -0.2, 0.4],
                    ],
                    dtype=torch.float32,
                ),
                expert_ids=torch.tensor(
                    [[0, 1, -1], [1, 0, 1], [0, -1, -1]],
                    dtype=torch.int32,
                ),
                routing_weights=torch.tensor(
                    [[0.35, 0.65, 0.0], [0.5, 0.2, 0.3], [0.8, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
            )

            async with asyncio.timeout(5):
                result = await execute_routed_layer(
                    batch,
                    _StaticTopology(topology),
                    RoundRobinSelector(),
                    pools,
                    monotonic_deadline=float("inf"),
                )

            torch.testing.assert_close(result, _reference(batch, weights))
            assert worker_a.ready.usage_count(0, 0) == 0
            assert worker_b.ready.usage_count(0, 1) == 0
            assert worker_a.server.active_count == worker_b.server.active_count == 0
        finally:
            for pool in pools.values():
                await pool.close()
            await asyncio.gather(*(worker.close() for worker in workers))

    asyncio.run(scenario())
