"""Tests for the synchronous framework bridge around the asyncio client."""

import asyncio
import concurrent.futures
import threading
from typing import ClassVar

import torch

from expertkit_transport.adapters.grpc import routed_client
from expertkit_transport.adapters.grpc.routed_client import BlockingGrpcRoutedMoEClient
from expertkit_transport.contracts import TransportError, TransportErrorCode


def client() -> BlockingGrpcRoutedMoEClient:
    return BlockingGrpcRoutedMoEClient(
        "127.0.0.1:50050",
        instance_id=7,
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=3,
        top_k=2,
        dtype=torch.float32,
        device="cpu",
    )


class ConcurrentAsyncClient:
    instances: ClassVar[list["ConcurrentAsyncClient"]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.loop_thread_id = threading.get_ident()
        self.entered = 0
        self.active = 0
        self.max_active = 0
        self.release = asyncio.Event()
        self.closed = False
        self.__class__.instances.append(self)

    async def start(self, *, monotonic_deadline: float) -> None:
        return None

    async def execute(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        distinct_expert_ids: tuple[int, ...],
        monotonic_deadline: float,
    ) -> torch.Tensor:
        assert threading.get_ident() == self.loop_thread_id
        assert distinct_expert_ids == (0, 1)
        self.entered += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.entered == 2:
            self.release.set()
        await self.release.wait()
        self.active -= 1
        return hidden_states + layer_id

    async def close(self) -> None:
        self.closed = True
        self.release.set()


def test_blocking_client_reuses_one_loop_and_allows_concurrent_calls(monkeypatch) -> None:
    ConcurrentAsyncClient.instances.clear()
    monkeypatch.setattr(routed_client, "GrpcRoutedMoEClient", ConcurrentAsyncClient)
    transport = client()
    transport.start(timeout_seconds=1)
    hidden = torch.tensor([[1.0, 2.0, 3.0]])
    experts = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    def execute(layer_id: int) -> torch.Tensor:
        return transport.execute(
            layer_id=layer_id,
            hidden_states=hidden,
            expert_ids=experts,
            routing_weights=weights,
            distinct_expert_ids=(0, 1),
            timeout_seconds=1,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(execute, 0)
        second = executor.submit(execute, 1)
        torch.testing.assert_close(first.result(), hidden)
        torch.testing.assert_close(second.result(), hidden + 1)

    fake = ConcurrentAsyncClient.instances[0]
    assert fake.max_active == 2
    assert fake.loop_thread_id != threading.get_ident()
    transport.close()
    assert fake.closed is True


class HangingAsyncClient:
    instances: ClassVar[list["HangingAsyncClient"]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.cancelled = threading.Event()
        self.__class__.instances.append(self)

    async def start(self, *, monotonic_deadline: float) -> None:
        return None

    async def execute(self, **kwargs) -> torch.Tensor:
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()

    async def close(self) -> None:
        return None


def test_blocking_timeout_waits_until_tensor_access_is_cancelled(monkeypatch) -> None:
    HangingAsyncClient.instances.clear()
    monkeypatch.setattr(routed_client, "GrpcRoutedMoEClient", HangingAsyncClient)
    transport = client()
    transport.start(timeout_seconds=1)

    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(max_workers=1):
        try:
            transport.execute(
                layer_id=0,
                hidden_states=torch.ones((1, 3)),
                expert_ids=torch.tensor([[0, 1]], dtype=torch.int32),
                routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                distinct_expert_ids=(0, 1),
                timeout_seconds=0.01,
            )
        except TransportError as error:
            assert error.code is TransportErrorCode.DEADLINE_EXCEEDED
        else:
            raise AssertionError("a hanging Routed-MoE call must time out")

    assert HangingAsyncClient.instances[0].cancelled.is_set()
    transport.close()


def test_blocking_client_rejects_calls_before_start(monkeypatch) -> None:
    monkeypatch.setattr(routed_client, "GrpcRoutedMoEClient", ConcurrentAsyncClient)
    transport = client()
    try:
        transport.execute(
            layer_id=0,
            hidden_states=torch.ones((1, 3)),
            expert_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            distinct_expert_ids=(0, 1),
            timeout_seconds=1,
        )
    except RuntimeError as error:
        assert "not running" in str(error)
    else:
        raise AssertionError("an unstarted client must reject execution")
    transport.close()
