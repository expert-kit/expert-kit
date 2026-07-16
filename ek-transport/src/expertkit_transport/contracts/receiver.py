"""Worker-side interface for taking admitted Transport batches."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from expertkit_transport.contracts.batches import WorkerBatch
from expertkit_transport.contracts.errors import TransportError


class ReceiverClosed(RuntimeError):
    """Indicate that Worker execution can no longer take Transport batches."""


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
    async def close(self) -> None:
        """Stop admission and release Transport resources."""
