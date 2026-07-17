"""Tests for the Torch model-facing Routed-MoE client."""

from typing import ClassVar

import pytest
import torch

from expertkit_torch import client
from expertkit_torch.client import RoutedMoEClient


class FakeTransport:
    instances: ClassVar[list["FakeTransport"]] = []

    def __init__(self, endpoint: str, **configuration: object) -> None:
        self.endpoint = endpoint
        self.configuration = configuration
        self.started_with: float | None = None
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def start(self, *, timeout_seconds: float) -> None:
        self.started_with = timeout_seconds

    def execute(self, **call: object) -> torch.Tensor:
        self.calls.append(call)
        return call["hidden_states"] + 1  # type: ignore[operator]

    def close(self) -> None:
        self.closed = True


def routed_client() -> RoutedMoEClient:
    return RoutedMoEClient(
        "127.0.0.1:50050",
        instance_id=7,
        num_layers=2,
        experts_per_layer=4,
        hidden_dim=3,
        top_k=2,
        timeout_seconds=2,
    )


def test_forwards_final_assignments_once_with_wire_dtypes(monkeypatch) -> None:
    FakeTransport.instances.clear()
    monkeypatch.setattr(client, "BlockingGrpcRoutedMoEClient", FakeTransport)
    routed = routed_client()
    hidden = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    expert_ids = torch.tensor([[1, 3]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float16)

    result = routed.forward_layer(
        layer_id=1,
        hidden_states=hidden,
        expert_ids=expert_ids,
        routing_weights=weights,
    )

    torch.testing.assert_close(result, hidden + 1)
    transport = FakeTransport.instances[0]
    assert transport.endpoint == "127.0.0.1:50050"
    assert transport.configuration["dtype"] is torch.float16
    assert transport.started_with == 2
    assert len(transport.calls) == 1
    assert transport.calls[0]["expert_ids"].dtype is torch.int32  # type: ignore[union-attr]
    assert transport.calls[0]["routing_weights"].dtype is torch.float32  # type: ignore[union-attr]
    routed.close()
    assert transport.closed is True


def test_reuses_one_transport_and_rejects_a_dtype_change(monkeypatch) -> None:
    FakeTransport.instances.clear()
    monkeypatch.setattr(client, "BlockingGrpcRoutedMoEClient", FakeTransport)
    routed = routed_client()
    hidden = torch.ones((1, 3), dtype=torch.float32)
    experts = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    for _ in range(2):
        routed.forward_layer(
            layer_id=0,
            hidden_states=hidden,
            expert_ids=experts,
            routing_weights=weights,
        )
    assert len(FakeTransport.instances) == 1

    with pytest.raises(ValueError, match="dtype changed"):
        routed.forward_layer(
            layer_id=0,
            hidden_states=hidden.to(torch.float16),
            expert_ids=experts,
            routing_weights=weights,
        )
    routed.close()


def test_rejects_mismatched_router_shapes_before_transport_submission(
    monkeypatch,
) -> None:
    FakeTransport.instances.clear()
    monkeypatch.setattr(client, "BlockingGrpcRoutedMoEClient", FakeTransport)
    routed = routed_client()

    with pytest.raises(ValueError, match="expert_ids"):
        routed.forward_layer(
            layer_id=0,
            hidden_states=torch.ones((2, 3)),
            expert_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            routing_weights=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        )
    assert FakeTransport.instances == []
    routed.close()


@pytest.mark.parametrize(
    ("expert_ids", "weights", "message"),
    [
        (
            torch.tensor([[2**32, 1]], dtype=torch.int64),
            torch.tensor([[0.5, 0.5]]),
            "configured expert range",
        ),
        (
            torch.tensor([[-2, 1]], dtype=torch.int64),
            torch.tensor([[0.5, 0.5]]),
            "configured expert range",
        ),
        (
            torch.tensor([[4, 1]], dtype=torch.int64),
            torch.tensor([[0.5, 0.5]]),
            "configured expert range",
        ),
        (
            torch.tensor([[1.9, 1.0]], dtype=torch.float32),
            torch.tensor([[0.5, 0.5]]),
            "integer dtype",
        ),
        (
            torch.tensor([[True, False]], dtype=torch.bool),
            torch.tensor([[0.5, 0.5]]),
            "integer dtype",
        ),
        (
            torch.tensor([[-1, 1]], dtype=torch.int64),
            torch.tensor([[0.5, 0.5]]),
            "zero routing weight",
        ),
    ],
)
def test_rejects_invalid_routing_before_starting_transport(
    monkeypatch,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    message: str,
) -> None:
    FakeTransport.instances.clear()
    monkeypatch.setattr(client, "BlockingGrpcRoutedMoEClient", FakeTransport)
    routed = routed_client()

    with pytest.raises(ValueError, match=message):
        routed.forward_layer(
            layer_id=0,
            hidden_states=torch.ones((1, 3)),
            expert_ids=expert_ids,
            routing_weights=weights,
        )

    assert FakeTransport.instances == []
    routed.close()


def test_forwards_valid_invalid_assignment_with_zero_weight(monkeypatch) -> None:
    FakeTransport.instances.clear()
    monkeypatch.setattr(client, "BlockingGrpcRoutedMoEClient", FakeTransport)
    routed = routed_client()

    routed.forward_layer(
        layer_id=0,
        hidden_states=torch.ones((1, 3)),
        expert_ids=torch.tensor([[-1, 3]], dtype=torch.int64),
        routing_weights=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )

    call = FakeTransport.instances[0].calls[0]
    assert call["expert_ids"].tolist() == [[-1, 3]]  # type: ignore[union-attr]
    assert call["expert_ids"].dtype is torch.int32  # type: ignore[union-attr]
    routed.close()
