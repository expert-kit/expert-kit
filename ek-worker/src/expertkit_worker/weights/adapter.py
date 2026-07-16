"""Backend-independent interface for expert weight conversion and sizing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from expertkit_worker.weights.format import SafeTensorData


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
