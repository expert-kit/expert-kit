"""End-to-end CPU execution through shared-memory Tensor slots."""

import asyncio
import time

import torch
from expertkit_transport.batches import WorkerBatch
from expertkit_transport.transports.base import WorkerPositionSpec
from expertkit_transport.transports.grpc import GrpcBatchSpec, GrpcWorkerServer
from expertkit_transport.transports.shm import ShmWorkerTransport

from expertkit_worker.backends.torch import TorchBackend, TorchExpertWeights
from expertkit_worker.execution import WorkerExecution
from expertkit_worker.weights import ReadyWeightTable

_HIDDEN_DIM = 4
_INTERMEDIATE_DIM = 6


def _weight(seed: int) -> TorchExpertWeights:
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> torch.Tensor:
        return (torch.randn(rows, columns, generator=generator) / 5).contiguous()

    return TorchExpertWeights(
        gate_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        up_proj=matrix(_INTERMEDIATE_DIM, _HIDDEN_DIM),
        down_proj=matrix(_HIDDEN_DIM, _INTERMEDIATE_DIM),
    )


def _reference(
    batch: WorkerBatch,
    weights: dict[int, TorchExpertWeights],
) -> torch.Tensor:
    result = torch.zeros_like(batch.hidden_states)
    for token_index in range(batch.token_count):
        token = batch.hidden_states[token_index]
        for route_index in range(batch.top_k):
            expert_id = int(batch.expert_ids[token_index, route_index])
            if expert_id < 0:
                continue
            weight = weights[expert_id]
            hidden = torch.nn.functional.silu(token @ weight.gate_proj.T)
            hidden.mul_(token @ weight.up_proj.T)
            result[token_index].add_(
                hidden @ weight.down_proj.T * batch.routing_weights[token_index, route_index]
            )
    return result


def test_worker_execution_uses_shared_input_and_output_destinations() -> None:
    async def scenario() -> None:
        batch_spec = GrpcBatchSpec(7, 1, 2, 4, _HIDDEN_DIM, 2, torch.float32)
        weights = {0: _weight(11), 1: _weight(29)}
        ready: ReadyWeightTable[TorchExpertWeights] = ReadyWeightTable(1, 2)
        for expert_id, weight in weights.items():
            ready.publish(0, expert_id, weight)
        server = GrpcWorkerServer(
            "127.0.0.1:0",
            batch_spec,
            max_active_batches=1,
            max_pending_batches=1,
        )
        backend = TorchBackend(
            hidden_dim=_HIDDEN_DIM,
            intermediate_dim=_INTERMEDIATE_DIM,
            top_k=2,
            dtype=torch.float32,
            device="cpu",
            acquire_many=ready.acquire_many,
        )
        execution = WorkerExecution(
            server,
            backend,
            instance_id=7,
            position_spec=WorkerPositionSpec(4, _HIDDEN_DIM, 2, torch.float32, "cpu"),
            active_positions=1,
        )
        await server.start()
        await execution.start()
        transport = ShmWorkerTransport(
            f"127.0.0.1:{server.bound_port}",
            batch_spec,
            max_in_flight=2,
            device="cpu",
        )
        await transport.start()
        output = torch.empty((2, _HIDDEN_DIM), dtype=torch.float32)
        batch = WorkerBatch(
            instance_id=7,
            layer_id=0,
            topology_version=1,
            hidden_states=torch.tensor(
                [[0.2, -0.4, 0.8, 0.5], [-0.1, 0.7, 0.3, -0.6]],
                dtype=torch.float32,
            ),
            token_indices=None,
            expert_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.35, 0.65], [0.8, 0.2]], dtype=torch.float32),
            distinct_expert_ids=(0, 1),
        )
        try:
            await transport.execute(
                batch,
                output,
                monotonic_deadline=time.monotonic() + 5,
            )
            torch.testing.assert_close(output, _reference(batch, weights))
        finally:
            await transport.close()
            await execution.close()

    asyncio.run(scenario())
