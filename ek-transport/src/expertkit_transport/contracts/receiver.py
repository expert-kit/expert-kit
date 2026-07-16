"""Worker-side interface for taking admitted Transport batches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from expertkit_transport.contracts.batches import WorkerBatch
from expertkit_transport.contracts.errors import TransportError


class ReceiverClosed(RuntimeError):
    """Indicate that Worker execution can no longer take Transport batches."""


@dataclass(frozen=True, slots=True)
class WorkerPositionSpec:
    """Describe the fixed Tensor storage owned by one active Worker position."""

    max_batch_tokens: int
    hidden_dim: int
    top_k: int
    dtype: torch.dtype
    device: torch.device | str

    def __post_init__(self) -> None:
        for name in ("max_batch_tokens", "hidden_dim", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be FP16, BF16, or FP32")
        device = torch.device(self.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Worker position device must be CPU or CUDA")
        object.__setattr__(self, "device", device)


class WorkerPositionBuffers(ABC):
    """Perform adapter-specific copies for one fixed active Worker position.

    The methods run in the Worker's bounded execution thread. For CUDA, the
    Worker selects the current stream before calling them.
    """

    @property
    @abstractmethod
    def host_staging_bytes(self) -> int:
        """Return fixed Host bytes allocated by this Transport adapter."""

    @abstractmethod
    def copy_input(
        self,
        batch: WorkerBatch,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        """Copy one received batch into valid views of fixed Backend inputs."""

    @abstractmethod
    def copy_output(self, partial_output: torch.Tensor) -> torch.Tensor:
        """Copy or expose a valid output view that this adapter can send."""

    @abstractmethod
    def close(self) -> None:
        """Release this position's adapter-specific fixed resources."""


class ReceivedWorkerBatch(ABC):
    """Represent one admitted batch until computation and response finish."""

    @property
    @abstractmethod
    def batch(self) -> WorkerBatch:
        """Return the validated Host batch that must enter an active position."""

    @property
    @abstractmethod
    def monotonic_deadline(self) -> float:
        """Return the absolute end-to-end deadline observed by Transport."""

    @property
    @abstractmethod
    def cancelled(self) -> bool:
        """Return whether the caller no longer needs a response."""

    @abstractmethod
    async def complete(self, partial_output: torch.Tensor) -> None:
        """Finish communication of one successful Weighted partial output."""

    @abstractmethod
    async def reject(self, error: TransportError) -> None:
        """Finish the request with a structured computation rejection."""


class WorkerBatchReceiver(ABC):
    """Supply admitted batches directly to Worker active positions."""

    @abstractmethod
    async def take(self) -> ReceivedWorkerBatch:
        """Wait for and remove the next batch from Transport-owned waiting data."""

    @abstractmethod
    def allocate_position_buffers(self, spec: WorkerPositionSpec) -> WorkerPositionBuffers:
        """Allocate adapter-specific fixed storage for one active position."""

    @abstractmethod
    async def close(self) -> None:
        """Stop admission and release Transport resources."""
