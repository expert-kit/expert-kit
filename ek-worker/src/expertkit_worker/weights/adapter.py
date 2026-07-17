"""Backend-independent interface for expert weight conversion and sizing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from expertkit_worker.weights.format import SafeTensorData


class WeightPlacementFatalReason(StrEnum):
    """Classify device failures that make continued Worker service unsafe."""

    DEVICE_OOM = "device_oom"
    DEVICE_FAILURE = "device_failure"


class WeightPlacementFatalError(RuntimeError):
    """Report a fatal failure while creating a final computation weight."""

    def __init__(self, reason: WeightPlacementFatalReason, diagnostic: str = "") -> None:
        super().__init__(diagnostic or reason.value)
        self.reason = reason
        self.diagnostic = diagnostic


class WeightAdapter[CpuWeightT, ReadyWeightT](ABC):
    """Convert validated expert files into one Backend's ready objects."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the configured built-in Backend name used for cache identity."""

    @abstractmethod
    def make_cpu_weight(self, source: SafeTensorData) -> CpuWeightT:
        """Create the cached CPU object from validated source regions."""

    @abstractmethod
    def make_ready_weight(self, cpu_weight: CpuWeightT) -> ReadyWeightT:
        """Create and fully synchronize the final computation object."""

    @abstractmethod
    def cpu_extra_bytes(self) -> int:
        """Return bytes allocated beyond the retained SafeTensors buffer."""

    @abstractmethod
    def source_tensor_bytes(self) -> int:
        """Return the exact encoded bytes for one expert's Tensor data."""

    @abstractmethod
    def ready_weight_bytes(self) -> int:
        """Return final device bytes for one equal-shaped expert."""

    @abstractmethod
    def conversion_temporary_bytes(self) -> int:
        """Return conservative temporary device bytes used during conversion."""

    def initialize_ready_storage(self, max_experts: int) -> None:
        """Initialize optional fixed Backend storage for the derived capacity.

        Backends whose ready objects own independent allocations use the default
        no-op. A fixed-slot Backend allocates its process-lifetime storage here,
        before the Worker registers with the Controller.
        """

        if isinstance(max_experts, bool) or not isinstance(max_experts, int) or max_experts <= 0:
            raise ValueError("max_experts must be a positive integer")

    def release_ready_weight(self, ready_weight: ReadyWeightT) -> None:
        """Release Backend-managed capacity after a ready object is withdrawn.

        This hook must not wait for device work. Weight Manager invokes it only
        after ready-weight usage reaches zero.
        """

        del ready_weight
