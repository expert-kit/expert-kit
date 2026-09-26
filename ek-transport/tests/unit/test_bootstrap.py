"""Deployment configuration stays inside EK and honors explicit injection."""

from __future__ import annotations

import asyncio
import json

import pytest
import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.bootstrap import runtimes_from_environment
from expertkit_transport.client import RoutedMoEClient


def test_missing_config_preserves_existing_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EK_TRANSPORT_CONFIG", raising=False)
    assert runtimes_from_environment(torch.device("cpu")) is None


def test_config_uses_client_device_and_creates_a_unique_runtime(tmp_path, monkeypatch) -> None:
    path = tmp_path / "transport.toml"
    path.write_text('[transfer_engine]\nsegment_name="127.0.0.1"\nprotocol="tcp"\n')
    monkeypatch.setenv("EK_TRANSPORT_CONFIG", str(path))
    first = runtimes_from_environment(torch.device("cpu"))
    second = runtimes_from_environment(torch.device("cpu"))
    key = lifecycle_pb2.WORKER_TRANSPORT_TRANSFER_ENGINE
    assert first[key].generation != second[key].generation
    assert first[key].device == torch.device("cpu")
    assert first[key].client_max_in_flight == 1
    asyncio.run(first[key].close())
    asyncio.run(second[key].close())


def test_config_cannot_override_the_framework_device(tmp_path, monkeypatch) -> None:
    path = tmp_path / "transport.json"
    path.write_text(json.dumps({"transfer_engine": {"device": "cuda:0"}}))
    monkeypatch.setenv("EK_TRANSPORT_CONFIG", str(path))
    with pytest.raises(ValueError, match="unsupported.*device"):
        runtimes_from_environment(torch.device("cpu"))


def test_explicit_runtime_bypasses_environment_configuration(monkeypatch) -> None:
    class Runtime:
        async def start(self):
            pass

        async def close(self):
            pass

    monkeypatch.setenv("EK_TRANSPORT_CONFIG", "/missing.json")
    runtime = Runtime()
    client = RoutedMoEClient(
        "127.0.0.1:19000",
        num_layers=1,
        experts_per_layer=2,
        hidden_dim=4,
        top_k=1,
        dtype=torch.float32,
        device="cpu",
        transport_runtime=runtime,
    )
    assert client._runtime_registry.runtime_for(4) is runtime
    asyncio.run(client.close())
