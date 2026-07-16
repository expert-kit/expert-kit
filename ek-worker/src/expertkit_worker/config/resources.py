"""Conservative startup calculation for device weight capacity."""

from __future__ import annotations

from dataclasses import dataclass

from expertkit_worker.backends import BackendResourceEstimate

_MINIMUM_ALLOCATOR_HEADROOM_BYTES = 512 * 1024 * 1024


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class DeviceResourcePlan:
    """Describe fixed non-weight reserve and remaining weight capacity."""

    device_memory_limit_bytes: int
    fixed_position_bytes: int
    backend_temporary_bytes: int
    conversion_temporary_bytes: int
    allocator_headroom_bytes: int
    weight_capacity_bytes: int

    @property
    def runtime_reserve_bytes(self) -> int:
        """Return every planned non-weight byte including allocator headroom."""

        return (
            self.fixed_position_bytes
            + self.backend_temporary_bytes
            + self.conversion_temporary_bytes
            + self.allocator_headroom_bytes
        )

    @property
    def bytes_to_allocate_after_fixed_positions(self) -> int:
        """Return planned future device use after fixed positions already exist."""

        return self.device_memory_limit_bytes - self.fixed_position_bytes


def plan_device_resources(
    *,
    device_memory_limit_bytes: int,
    fixed_position_bytes: int,
    backend_estimate: BackendResourceEstimate,
    active_batches: int,
    conversion_temporary_bytes: int,
) -> DeviceResourcePlan:
    """Derive fixed weight capacity from the selected runtime configuration.

    Raises:
        ValueError: The configured device limit cannot reserve one positive byte
            for expert weights after conservative runtime requirements.
    """

    _positive_int("device_memory_limit_bytes", device_memory_limit_bytes)
    _nonnegative_int("fixed_position_bytes", fixed_position_bytes)
    _positive_int("active_batches", active_batches)
    _nonnegative_int("conversion_temporary_bytes", conversion_temporary_bytes)
    if not isinstance(backend_estimate, BackendResourceEstimate):
        raise TypeError("backend_estimate must be a BackendResourceEstimate")

    backend_temporary_bytes = backend_estimate.total_bytes(active_batches)
    allocator_headroom_bytes = max(
        device_memory_limit_bytes // 10,
        _MINIMUM_ALLOCATOR_HEADROOM_BYTES,
    )
    runtime_reserve_bytes = (
        fixed_position_bytes
        + backend_temporary_bytes
        + conversion_temporary_bytes
        + allocator_headroom_bytes
    )
    weight_capacity_bytes = device_memory_limit_bytes - runtime_reserve_bytes
    if weight_capacity_bytes <= 0:
        raise ValueError("device_memory_limit cannot fit runtime reserve and expert weights")

    return DeviceResourcePlan(
        device_memory_limit_bytes=device_memory_limit_bytes,
        fixed_position_bytes=fixed_position_bytes,
        backend_temporary_bytes=backend_temporary_bytes,
        conversion_temporary_bytes=conversion_temporary_bytes,
        allocator_headroom_bytes=allocator_headroom_bytes,
        weight_capacity_bytes=weight_capacity_bytes,
    )


def validate_available_device_memory(
    plan: DeviceResourcePlan,
    *,
    available_bytes_after_fixed_positions: int,
) -> None:
    """Reject startup when current free device memory cannot honor the plan."""

    if not isinstance(plan, DeviceResourcePlan):
        raise TypeError("plan must be a DeviceResourcePlan")
    _nonnegative_int(
        "available_bytes_after_fixed_positions",
        available_bytes_after_fixed_positions,
    )
    required = plan.bytes_to_allocate_after_fixed_positions
    if available_bytes_after_fixed_positions < required:
        raise ValueError(
            "available device memory is smaller than the configured post-allocation plan"
        )
