"""Prepared-output ownership contract used by Transport implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from expertkit_transport.batches import ACTIVATION_DTYPES


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Describe one reusable maximum-size Frontend output buffer."""

    max_batch_tokens: int
    hidden_dim: int
    dtype: torch.dtype
    device: torch.device | str

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_batch_tokens, bool)
            or not isinstance(self.max_batch_tokens, int)
            or self.max_batch_tokens <= 0
        ):
            raise ValueError("max_batch_tokens must be positive")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be positive")
        if self.dtype not in ACTIVATION_DTYPES:
            raise ValueError("output dtype must be FP16, BF16, or FP32")
        object.__setattr__(self, "device", torch.device(self.device))


class PreparedOutput(ABC):
    """Hold a reusable Tensor and any adapter-private allocation state."""

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Return the maximum-size Tensor exposed to middleware."""


class OutputBufferProvider(ABC):
    """Allocate and release outputs compatible with one Transport adapter."""

    @abstractmethod
    def prepare(self, spec: OutputSpec) -> PreparedOutput:
        """Allocate and, when necessary, register one reusable output."""

    @abstractmethod
    def validate(self, output: PreparedOutput, spec: OutputSpec) -> None:
        """Reject an output that cannot receive results for `spec`."""

    @abstractmethod
    def before_receive(self, output: PreparedOutput) -> None:
        """Order the adapter's actual receive operation after prior consumption."""

    @abstractmethod
    def after_consume(self, output: PreparedOutput) -> None:
        """Record downstream work that must finish before output reuse."""

    @abstractmethod
    def release(self, output: PreparedOutput) -> None:
        """Release adapter-owned registration or allocation state."""
