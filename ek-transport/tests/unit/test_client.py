"""Tests for high-level routed-layer client startup."""

import asyncio
import time
from typing import ClassVar

import torch

import expertkit_transport.client as client_module
from expertkit_transport.client import RoutedMoEClient
from expertkit_transport.controller import ResolvedDefaultInstance


class FakeTopology:
    instances: ClassVar[list["FakeTopology"]] = []

    def __init__(self, endpoint: str, **configuration: object) -> None:
        self.endpoint = endpoint
        self.configuration = configuration
        self.pools: dict[object, object] = {}
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    async def start(self, *, monotonic_deadline: float) -> None:
        assert monotonic_deadline > time.monotonic()
        self.started = True

    async def close(self) -> None:
        self.closed = True


def test_client_resolves_default_before_constructing_instance_bound_topology(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        requests: list[int | None] = []
        submitted: list[object] = []

        async def resolve(endpoint, *, requested_instance_id, timeout_seconds):
            assert endpoint == "controller:5002"
            assert timeout_seconds > 0
            requests.append(requested_instance_id)
            return ResolvedDefaultInstance(7, "model", "default")

        async def execute(batch, *args, **kwargs):
            submitted.append(batch)
            return batch.hidden_states

        FakeTopology.instances.clear()
        monkeypatch.setattr(client_module, "resolve_default_instance", resolve)
        monkeypatch.setattr(client_module, "ControllerTopologyWatcher", FakeTopology)
        monkeypatch.setattr(client_module, "execute_routed_layer", execute)
        client = RoutedMoEClient(
            "controller:5002",
            num_layers=2,
            experts_per_layer=4,
            hidden_dim=3,
            top_k=2,
            dtype=torch.float32,
            device="cpu",
        )

        await client.start(monotonic_deadline=time.monotonic() + 1)
        hidden = torch.ones((1, 3))
        result = await client.execute(
            layer_id=1,
            hidden_states=hidden,
            expert_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            distinct_expert_ids=(0, 1),
            monotonic_deadline=time.monotonic() + 1,
        )

        assert result is hidden
        assert requests == [None]
        assert FakeTopology.instances[0].configuration["instance_id"] == 7
        assert submitted[0].instance_id == 7
        await client.close()
        assert FakeTopology.instances[0].closed is True

    asyncio.run(scenario())
