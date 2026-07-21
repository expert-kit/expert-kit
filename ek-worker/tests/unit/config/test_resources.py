"""Tests for conservative Worker device-memory planning."""

import pytest

from expertkit_worker.backends import BackendResourceEstimate
from expertkit_worker.config.resources import (
    plan_device_resources,
    validate_available_device_memory,
)

_GIB = 1024 * 1024 * 1024
_MIB = 1024 * 1024


def test_plan_reserves_fixed_temporary_conversion_and_headroom_bytes() -> None:
    plan = plan_device_resources(
        device_memory_limit_bytes=8 * _GIB,
        fixed_slot_bytes=100 * _MIB,
        backend_estimate=BackendResourceEstimate(
            temporary_bytes_per_active_batch=200 * _MIB,
            shared_temporary_bytes=50 * _MIB,
        ),
        active_batches=2,
        conversion_temporary_bytes=25 * _MIB,
    )

    assert plan.backend_temporary_bytes == 450 * _MIB
    assert plan.allocator_headroom_bytes == (8 * _GIB) // 10
    assert plan.runtime_reserve_bytes == (100 * _MIB + 450 * _MIB + 25 * _MIB + (8 * _GIB) // 10)
    assert plan.weight_capacity_bytes == 8 * _GIB - plan.runtime_reserve_bytes
    assert plan.bytes_to_allocate_after_fixed_slots == 8 * _GIB - 100 * _MIB


def test_plan_uses_minimum_headroom_and_rejects_exhausted_limit() -> None:
    plan = plan_device_resources(
        device_memory_limit_bytes=2 * _GIB,
        fixed_slot_bytes=1,
        backend_estimate=BackendResourceEstimate(0),
        active_batches=1,
        conversion_temporary_bytes=0,
    )
    assert plan.allocator_headroom_bytes == 512 * _MIB

    with pytest.raises(ValueError, match="cannot fit"):
        plan_device_resources(
            device_memory_limit_bytes=512 * _MIB,
            fixed_slot_bytes=1,
            backend_estimate=BackendResourceEstimate(0),
            active_batches=1,
            conversion_temporary_bytes=0,
        )


def test_current_free_memory_must_cover_every_planned_future_allocation() -> None:
    plan = plan_device_resources(
        device_memory_limit_bytes=2 * _GIB,
        fixed_slot_bytes=64 * _MIB,
        backend_estimate=BackendResourceEstimate(0),
        active_batches=1,
        conversion_temporary_bytes=0,
    )
    validate_available_device_memory(
        plan,
        available_bytes_after_fixed_slots=plan.bytes_to_allocate_after_fixed_slots,
    )
    with pytest.raises(ValueError, match="available device memory"):
        validate_available_device_memory(
            plan,
            available_bytes_after_fixed_slots=(plan.bytes_to_allocate_after_fixed_slots - 1),
        )
