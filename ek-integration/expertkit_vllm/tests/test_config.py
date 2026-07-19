"""Tests for vLLM process configuration."""

import pytest
from expertkit_vllm.utils.config import collect_ek_client_config


def test_collects_numeric_instance_and_timeout(monkeypatch) -> None:
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    monkeypatch.setenv("EK_CLIENT_TIMEOUT", "2.5")

    config = collect_ek_client_config()

    assert config.controller_endpoint == "127.0.0.1:50050"
    assert config.instance_id == 7
    assert config.timeout_seconds == 2.5
    assert config.transport == "grpc"


def test_selects_same_host_shared_memory(monkeypatch) -> None:
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    monkeypatch.setenv("EK_WORKER_TRANSPORT", "shm")

    assert collect_ek_client_config().transport == "shm"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("EK_INSTANCE_ID", "0", "positive"),
        ("EK_INSTANCE_ID", "worker", "integer"),
        ("EK_CLIENT_TIMEOUT", "nan", "finite"),
        ("EK_CLIENT_TIMEOUT", "0", "positive"),
        ("EK_WORKER_TRANSPORT", "rdma", "grpc.*shm"),
    ],
)
def test_rejects_invalid_values(
    monkeypatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        collect_ek_client_config()


def test_requires_an_instance_id(monkeypatch) -> None:
    monkeypatch.delenv("EK_INSTANCE_ID", raising=False)
    with pytest.raises(ValueError, match="must be set"):
        collect_ek_client_config()
