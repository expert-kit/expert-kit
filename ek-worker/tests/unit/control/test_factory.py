"""Tests for constructing one coherent Controller process identity."""

from types import SimpleNamespace

import torch

from expertkit_worker.control.factory import create_controller_supervisor


def test_controller_components_share_the_caller_supplied_start_id() -> None:
    config = SimpleNamespace(
        controller=SimpleNamespace(
            endpoint="controller:50050",
            heartbeat_interval_secs=3,
            heartbeat_timeout_secs=10,
        ),
        worker=SimpleNamespace(
            id="worker-0",
            backend=SimpleNamespace(value="torch"),
            device="cuda:0",
            max_batch_tokens=16,
            max_active_batches_per_device=1,
            shutdown_grace_secs=30,
        ),
        transport=SimpleNamespace(max_pending_batches_per_device=2),
        weight_manager=SimpleNamespace(peer=SimpleNamespace(advertise="http://worker:50052")),
    )
    manager = SimpleNamespace(max_experts=8)

    supervisor = create_controller_supervisor(
        config,  # type: ignore[arg-type]
        instance_id=7,
        activation_dtype=torch.float16,
        receiver=object(),  # type: ignore[arg-type]
        manager=manager,  # type: ignore[arg-type]
        reporter=object(),  # type: ignore[arg-type]
        computation_endpoint="worker:50051",
        transport_type="transfer_engine",
        start_id="start-shared-by-receiver-and-controller",
    )

    assert supervisor._registration.start_id == "start-shared-by-receiver-and-controller"
    assert supervisor._heartbeat._start_id == "start-shared-by-receiver-and-controller"
    assert supervisor._weights._start_id == "start-shared-by-receiver-and-controller"
