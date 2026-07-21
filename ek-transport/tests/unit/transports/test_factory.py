"""Tests for selecting a Frontend Worker Transport from Controller topology."""

import torch
from expertkit_proto.ek.control.v2 import lifecycle_pb2

from expertkit_transport.transports import factory


class FakeTransport:
    """Capture concrete Transport constructor arguments without opening resources."""

    def __init__(self, endpoint: str, batch_spec: object, **options: object) -> None:
        self.endpoint = endpoint
        self.batch_spec = batch_spec
        self.options = options


def create(transport_type: int) -> FakeTransport:
    return factory.create_worker_transport(  # type: ignore[return-value]
        transport_type=transport_type,
        endpoint="worker:50052",
        instance_id=7,
        num_layers=4,
        experts_per_layer=8,
        max_batch_tokens=16,
        hidden_dim=32,
        top_k=2,
        dtype=torch.float16,
        device=torch.device("cpu"),
        max_in_flight=3,
    )


def test_creates_grpc_transport_from_published_type(monkeypatch) -> None:
    monkeypatch.setattr(factory, "GrpcWorkerTransport", FakeTransport)

    transport = create(lifecycle_pb2.WORKER_TRANSPORT_GRPC)

    assert transport.endpoint == "worker:50052"
    assert transport.batch_spec.instance_id == 7
    assert transport.options == {"max_in_flight": 3}


def test_creates_shm_transport_from_published_type(monkeypatch) -> None:
    monkeypatch.setattr(factory, "ShmWorkerTransport", FakeTransport)

    transport = create(lifecycle_pb2.WORKER_TRANSPORT_SHM)

    assert transport.endpoint == "worker:50052"
    assert transport.options == {
        "max_in_flight": 3,
        "device": torch.device("cpu"),
    }


def test_rejects_an_unknown_published_type() -> None:
    try:
        create(999)
    except ValueError as error:
        assert "unsupported Worker transport type" in str(error)
    else:
        raise AssertionError("an unknown Transport type must be rejected")
