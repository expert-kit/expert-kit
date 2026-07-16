"""Protocol tests for v2 lifecycle, Topology, and weight-control messages."""

from expertkit_transport._proto.ek.control.v2 import (
    lifecycle_pb2,
    lifecycle_pb2_grpc,
    weight_control_pb2,
    weight_control_pb2_grpc,
)
from expertkit_transport._proto.ek.worker.v2 import common_pb2


def test_control_service_streaming_shapes_are_stable() -> None:
    lifecycle = lifecycle_pb2.DESCRIPTOR.services_by_name["WorkerLifecycleService"]
    register = lifecycle.methods_by_name["RegisterWorker"]
    heartbeat = lifecycle.methods_by_name["Heartbeat"]
    topology = lifecycle_pb2.DESCRIPTOR.services_by_name["TopologyService"].methods_by_name[
        "WatchTopology"
    ]
    weight = weight_control_pb2.DESCRIPTOR.services_by_name["WeightControlService"].methods_by_name[
        "Sync"
    ]

    assert (register.client_streaming, register.server_streaming) == (False, False)
    assert (heartbeat.client_streaming, heartbeat.server_streaming) == (True, False)
    assert (topology.client_streaming, topology.server_streaming) == (False, True)
    assert (weight.client_streaming, weight.server_streaming) == (True, True)
    assert hasattr(lifecycle_pb2_grpc, "WorkerLifecycleServiceStub")
    assert hasattr(lifecycle_pb2_grpc, "TopologyServiceStub")
    assert hasattr(weight_control_pb2_grpc, "WeightControlServiceStub")


def test_registration_round_trip_uses_one_device() -> None:
    request = lifecycle_pb2.RegisterWorkerRequest(
        worker_id="worker-0",
        start_id="start-1",
        instance_id=7,
        computation_endpoint="worker-0:50051",
        peer_weight_endpoint="http://worker-0:50052",
        backend="torch",
        activation_dtype=common_pb2.ACTIVATION_DTYPE_BF16,
        device=lifecycle_pb2.WorkerDevice(device="cuda:0", max_experts=8),
        max_batch_tokens=4096,
        max_active_batches_per_device=1,
        max_pending_batches_per_device=1,
    )

    decoded = lifecycle_pb2.RegisterWorkerRequest.FromString(request.SerializeToString())

    assert decoded == request
    assert decoded.device.device == "cuda:0"
    assert decoded.device.max_experts == 8


def test_topology_message_keeps_snapshot_and_update_exclusive() -> None:
    message = lifecycle_pb2.TopologyMessage(
        snapshot=lifecycle_pb2.TopologySnapshotPart(
            instance_id=7,
            topology_version=11,
            part_index=0,
            part_count=1,
            routes=[
                lifecycle_pb2.ExpertRoute(
                    layer_id=2,
                    expert_id=3,
                    replicas=[
                        lifecycle_pb2.WorkerRoute(
                            worker_id="worker-0",
                            start_id="start-1",
                            computation_endpoint="worker-0:50051",
                            device="cuda:0",
                            max_active_batches=1,
                            max_pending_batches=1,
                        )
                    ],
                )
            ],
        )
    )
    assert message.WhichOneof("message") == "snapshot"

    message.update.CopyFrom(
        lifecycle_pb2.TopologyUpdatePart(
            instance_id=7,
            previous_version=11,
            topology_version=12,
            part_index=0,
            part_count=1,
            changes=[lifecycle_pb2.RouteChange(layer_id=2, expert_id=3)],
        )
    )

    assert message.WhichOneof("message") == "update"
    assert len(message.update.changes[0].replicas) == 0


def test_weight_stream_envelopes_round_trip() -> None:
    worker_message = weight_control_pb2.WorkerWeightMessage(
        state_updates=weight_control_pb2.ExpertStateUpdates(
            placement_generation=4,
            report_sequence=9,
            experts=[
                weight_control_pb2.ExpertState(
                    layer_id=2,
                    expert_id=3,
                    state=weight_control_pb2.EXPERT_FAILED,
                    failure=weight_control_pb2.WeightLoadFailure(
                        stage=weight_control_pb2.WEIGHT_LOAD_VALIDATE,
                        code=weight_control_pb2.WEIGHT_LOAD_ERROR_UNEXPECTED_METADATA,
                        retryable=False,
                        diagnostic="shape mismatch",
                    ),
                )
            ],
        )
    )
    decoded_worker = weight_control_pb2.WorkerWeightMessage.FromString(
        worker_message.SerializeToString()
    )
    assert decoded_worker.WhichOneof("message") == "state_updates"
    assert decoded_worker.state_updates.report_sequence == 9

    controller_message = weight_control_pb2.ControllerWeightMessage(
        drain=weight_control_pb2.DrainAuthorizationPart(
            drain_id=5,
            placement_generation=4,
            min_topology_version=12,
            stop_accepting_all_computation=True,
            part_index=0,
            part_count=1,
            experts=[common_pb2.ExpertKey(layer_id=2, expert_id=3)],
        )
    )
    decoded_controller = weight_control_pb2.ControllerWeightMessage.FromString(
        controller_message.SerializeToString()
    )
    assert decoded_controller.WhichOneof("message") == "drain"
    assert decoded_controller.drain.experts[0].expert_id == 3


def test_empty_target_list_is_one_explicit_part() -> None:
    message = weight_control_pb2.ControllerWeightMessage(
        targets=weight_control_pb2.TargetExpertListPart(
            placement_generation=6,
            part_index=0,
            part_count=1,
        )
    )

    assert message.WhichOneof("message") == "targets"
    assert message.targets.part_count == 1
    assert len(message.targets.experts) == 0
