"""Real two-rank NCCL smoke test; skipped without two CUDA devices."""

from __future__ import annotations

import asyncio
import socket
import time
from multiprocessing.queues import SimpleQueue

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.transports.base import BatchBufferConfig, WorkerEndpointConfig
from expertkit_transport.transports.nccl import (
    NcclRuntime,
    NcclRuntimeConfig,
    NcclWorkerBatchReceiver,
    NcclWorkerTransport,
)


def _nccl_is_available() -> bool:
    try:
        return torch.cuda.device_count() >= 2 and dist.is_available() and dist.is_nccl_available()
    except (AttributeError, RuntimeError):
        return False


def _free_tcp_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _run_nccl_rank(
    rank: int,
    world_size: int,
    rendezvous_endpoint: str,
    results: SimpleQueue,
) -> None:
    async def scenario() -> None:
        device = torch.device(f"cuda:{rank}")
        runtime = NcclRuntime(
            NcclRuntimeConfig(
                rank=rank,
                world_size=world_size,
                rendezvous_endpoint=rendezvous_endpoint,
                group_name="expertkit-test",
                device=device,
                init_timeout_seconds=30,
            )
        )
        deadline = time.monotonic() + 30
        await runtime.start()
        await runtime.wait_ready(monotonic_deadline=deadline)
        consumer_stream = torch.cuda.Stream(device=device)
        try:
            if rank == 0:
                hidden_states = torch.tensor(
                    [[1, 2, 3], [4, 5, 6]],
                    dtype=torch.float32,
                    device=device,
                )
                expert_ids = torch.tensor(
                    [[1, -1], [0, 3]],
                    dtype=torch.int32,
                    device=device,
                )
                routing_weights = torch.tensor(
                    [[0.25, 0], [0.5, 0.5]],
                    dtype=torch.float32,
                    device=device,
                )
                output = torch.empty_like(hidden_states)
                await runtime.exchange(
                    1,
                    hidden_states,
                    expert_ids,
                    routing_weights,
                    output,
                    monotonic_deadline=deadline,
                )
                with torch.cuda.stream(consumer_stream):
                    observed = output.clone()
                consumer_stream.synchronize()
                results.put(observed.cpu().tolist())
            else:
                hidden_states = torch.empty((2, 3), dtype=torch.float32, device=device)
                expert_ids = torch.empty((2, 2), dtype=torch.int32, device=device)
                routing_weights = torch.empty((2, 2), dtype=torch.float32, device=device)
                dummy_output = torch.zeros_like(hidden_states)
                exchange = await runtime.receive_inputs(
                    0,
                    hidden_states,
                    expert_ids,
                    routing_weights,
                    dummy_output,
                    monotonic_deadline=deadline,
                )
                with torch.cuda.stream(consumer_stream):
                    observed_expert_ids = expert_ids.clone()
                    response = hidden_states * 2
                consumer_stream.synchronize()
                torch.testing.assert_close(
                    observed_expert_ids,
                    torch.tensor([[1, -1], [0, 3]], dtype=torch.int32, device=device),
                )
                await exchange.send_output(
                    response,
                    monotonic_deadline=deadline,
                )
        finally:
            await runtime.close()

    asyncio.run(scenario())


def _run_nccl_transport_rank(
    rank: int,
    world_size: int,
    rendezvous_endpoint: str,
    control_endpoint: str,
    worker_ready: object,
    client_done: object,
    results: SimpleQueue,
) -> None:
    async def scenario() -> None:
        device = torch.device(f"cuda:{rank}")
        runtime = NcclRuntime(
            NcclRuntimeConfig(
                rank=rank,
                world_size=world_size,
                rendezvous_endpoint=rendezvous_endpoint,
                group_name="expertkit-transport-test",
                device=device,
                init_timeout_seconds=30,
            )
        )
        endpoint = WorkerEndpointConfig(
            instance_id=7,
            num_layers=2,
            experts_per_layer=4,
            max_batch_tokens=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float32,
        )
        consumer_stream = torch.cuda.Stream(device=device)
        if rank == 0:
            assert await asyncio.to_thread(worker_ready.wait, 10)
            transport = NcclWorkerTransport(
                control_endpoint,
                endpoint,
                max_in_flight=1,
                device=device,
                runtime=runtime,
                peer_rank=1,
            )
            try:
                await transport.start()
                batch = WorkerBatch(
                    instance_id=7,
                    layer_id=1,
                    topology_version=3,
                    hidden_states=torch.tensor(
                        [[1, 2, 3], [4, 5, 6]],
                        dtype=torch.float32,
                        device=device,
                    ),
                    token_indices=None,
                    expert_ids=torch.tensor(
                        [[0, 1], [2, -1]],
                        dtype=torch.int32,
                        device=device,
                    ),
                    routing_weights=torch.tensor(
                        [[0.25, 0.75], [1, 0]],
                        dtype=torch.float32,
                        device=device,
                    ),
                    distinct_expert_ids=(0, 1, 2),
                )
                output = torch.empty_like(batch.hidden_states)
                await transport.execute(
                    batch,
                    output,
                    monotonic_deadline=time.monotonic() + 30,
                )
                with torch.cuda.stream(consumer_stream):
                    observed = output.clone()
                consumer_stream.synchronize()
                results.put(observed.cpu().tolist())
                client_done.set()
            finally:
                await transport.close()
                await runtime.close()
            return

        receiver = NcclWorkerBatchReceiver(
            control_endpoint,
            endpoint,
            runtime=runtime,
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=True,
        )
        execution_buffers = receiver.create_batch_buffers(
            BatchBufferConfig(4, 3, 2, torch.float32, device)
        )
        try:
            await receiver.start()
            worker_ready.set()
            received = await receiver.receive()
            destination = received.output_destination
            assert destination is not None
            with torch.cuda.stream(consumer_stream):
                destination.copy_(received.batch.hidden_states * 2)
            consumer_stream.synchronize()
            received.release_input()
            await received.complete(destination)
            assert await asyncio.to_thread(client_done.wait, 10)
        finally:
            await receiver.close()
            execution_buffers.close()

    asyncio.run(scenario())


@pytest.mark.cuda
@pytest.mark.skipif(
    not _nccl_is_available(),
    reason="requires two CUDA devices and a PyTorch NCCL build",
)
def test_two_processes_exchange_tensors_with_real_nccl() -> None:
    context = mp.get_context("spawn")
    results = context.SimpleQueue()
    mp.spawn(
        _run_nccl_rank,
        args=(2, _free_tcp_endpoint(), results),
        nprocs=2,
        join=True,
    )

    assert results.get() == [[2, 4, 6], [8, 10, 12]]


@pytest.mark.cuda
@pytest.mark.skipif(
    not _nccl_is_available(),
    reason="requires two CUDA devices and a PyTorch NCCL build",
)
def test_two_processes_execute_over_nccl_transport() -> None:
    context = mp.get_context("spawn")
    results = context.SimpleQueue()
    worker_ready = context.Event()
    client_done = context.Event()
    mp.spawn(
        _run_nccl_transport_rank,
        args=(
            2,
            _free_tcp_endpoint(),
            _free_tcp_endpoint().removeprefix("tcp://"),
            worker_ready,
            client_done,
            results,
        ),
        nprocs=2,
        join=True,
    )

    assert results.get() == [[2, 4, 6], [8, 10, 12]]
