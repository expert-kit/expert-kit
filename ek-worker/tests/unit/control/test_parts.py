"""Tests for atomic multipart weight-control message assembly."""

import pytest
from expertkit_proto.ek.control.v2 import weight_control_pb2
from expertkit_proto.ek.worker.v2 import common_pb2

from expertkit_worker.control import DrainAuthorizationAssembler, TargetListAssembler
from expertkit_worker.weights.dram_cache import WeightKey


def _target_part(
    generation: int,
    part_index: int,
    part_count: int,
    expert_ids: tuple[int, ...],
) -> weight_control_pb2.TargetExpertListPart:
    return weight_control_pb2.TargetExpertListPart(
        placement_generation=generation,
        part_index=part_index,
        part_count=part_count,
        experts=[
            weight_control_pb2.TargetExpert(
                layer_id=2,
                expert_id=expert_id,
                target_device="cuda:0",
                peer_weight_endpoints=[f"http://peer-{expert_id}:8080"],
            )
            for expert_id in expert_ids
        ],
    )


def _drain_part(
    drain_id: int,
    part_index: int,
    part_count: int,
    expert_ids: tuple[int, ...],
    *,
    generation: int = 7,
    topology_version: int = 11,
    stop_all: bool = False,
) -> weight_control_pb2.DrainAuthorizationPart:
    return weight_control_pb2.DrainAuthorizationPart(
        drain_id=drain_id,
        placement_generation=generation,
        min_topology_version=topology_version,
        stop_accepting_all_computation=stop_all,
        part_index=part_index,
        part_count=part_count,
        experts=[common_pb2.ExpertKey(layer_id=2, expert_id=value) for value in expert_ids],
    )


def test_target_assembler_applies_out_of_order_parts_atomically() -> None:
    assembler = TargetListAssembler(max_experts=4)
    second = _target_part(5, 1, 2, (3,))
    first = _target_part(5, 0, 2, (1, 2))

    assert assembler.add(second) is None
    complete = assembler.add(first)

    assert complete is not None
    assert complete.placement_generation == 5
    assert [expert.expert_id for expert in complete.experts] == [1, 2, 3]
    assert complete.experts[0].peer_endpoints == ("http://peer-1:8080",)
    assert assembler.add(first) == complete


def test_newer_target_generation_discards_incomplete_older_parts() -> None:
    assembler = TargetListAssembler(max_experts=4)
    assert assembler.add(_target_part(3, 0, 2, (0,))) is None

    newest = assembler.add(_target_part(4, 0, 1, (2,)))

    assert newest is not None
    assert newest.placement_generation == 4
    assert assembler.add(_target_part(3, 1, 2, (1,))) is None
    assert assembler.add(_target_part(3, 99, 0, tuple(range(65)))) is None


def test_empty_target_list_requires_one_explicit_part() -> None:
    assembler = TargetListAssembler(max_experts=1)

    complete = assembler.add(_target_part(1, 0, 1, ()))

    assert complete is not None
    assert complete.experts == ()


def test_target_assembler_rejects_conflicts_and_duplicate_experts() -> None:
    assembler = TargetListAssembler(max_experts=4)
    first = _target_part(2, 0, 2, (1,))
    assert assembler.add(first) is None

    with pytest.raises(ValueError, match="different content"):
        assembler.add(_target_part(2, 0, 2, (2,)))
    with pytest.raises(ValueError, match="duplicate expert"):
        assembler.add(_target_part(2, 1, 2, (1,)))


def test_target_assembler_enforces_part_and_total_bounds() -> None:
    assembler = TargetListAssembler(max_experts=64)

    with pytest.raises(ValueError, match="more than 64"):
        assembler.add(_target_part(1, 0, 1, tuple(range(65))))
    with pytest.raises(ValueError, match="part_index"):
        assembler.add(_target_part(1, 1, 1, (0,)))
    with pytest.raises(ValueError, match="part_count"):
        assembler.add(_target_part(1, 0, 65, (0,)))


def test_drain_assembler_supports_interleaving_and_exact_repeats() -> None:
    assembler = DrainAuthorizationAssembler(max_experts=4)
    drain_one_last = _drain_part(10, 1, 2, (3,))
    drain_two = _drain_part(11, 0, 1, (2,), generation=8, topology_version=12)

    assert assembler.add(drain_one_last) is None
    assert assembler.add(drain_one_last) is None
    second = assembler.add(drain_two)
    first = assembler.add(_drain_part(10, 0, 2, (1,)))

    assert second is not None
    assert second.drain_id == 11
    assert second.experts == (WeightKey(2, 2),)
    assert first is not None
    assert first.drain_id == 10
    assert first.experts == (WeightKey(2, 1), WeightKey(2, 3))


def test_drain_assembler_rejects_conflicting_metadata_and_duplicate_experts() -> None:
    assembler = DrainAuthorizationAssembler(max_experts=4)
    assert assembler.add(_drain_part(3, 0, 2, (1,))) is None

    with pytest.raises(ValueError, match="metadata"):
        assembler.add(_drain_part(3, 1, 2, (2,), topology_version=99))
    with pytest.raises(ValueError, match="duplicate expert"):
        assembler.add(_drain_part(3, 1, 2, (1,)))


def test_empty_drain_is_valid_only_for_whole_worker_shutdown() -> None:
    assembler = DrainAuthorizationAssembler(max_experts=1)

    with pytest.raises(ValueError, match="must name"):
        assembler.add(_drain_part(4, 0, 1, ()))
    complete = assembler.add(_drain_part(5, 0, 1, (), stop_all=True))

    assert complete is not None
    assert complete.stop_accepting_all_computation is True
    assert complete.experts == ()


def test_drain_assembler_bounds_incomplete_authorizations() -> None:
    assembler = DrainAuthorizationAssembler(max_experts=64)
    for drain_id in range(1, 65):
        assert assembler.add(_drain_part(drain_id, 0, 2, (0,))) is None

    with pytest.raises(ValueError, match="too many incomplete"):
        assembler.add(_drain_part(65, 0, 2, (0,)))
