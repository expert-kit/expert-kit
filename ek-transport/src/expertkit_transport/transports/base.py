"""Worker-side interface for taking admitted Transport batches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.errors import TransportError
from expertkit_transport.tracing import TraceContext

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


def _require_positive_unsigned(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be a positive integer no larger than {maximum}")


class WorkerTransport(ABC):
    """Submit Worker batches to one remote Worker."""

    @abstractmethod
    async def start(self) -> None:
        """Create Transport resources on the current event loop."""

    @abstractmethod
    async def execute(
        self,
        batch: WorkerBatch,
        output: torch.Tensor,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Fill a caller-owned output Tensor or raise a Transport error."""

    @abstractmethod
    async def close(self) -> None:
        """Stop submissions and release connection resources."""


class ReceiverClosed(RuntimeError):
    """Indicate that Worker execution can no longer take Transport batches."""


@dataclass(frozen=True, slots=True)
class WorkerEndpointConfig:
    """Fix the model identity, shape, and dtype accepted by one Worker endpoint."""

    instance_id: int
    num_layers: int
    experts_per_layer: int
    max_batch_tokens: int
    hidden_dim: int
    top_k: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        _require_positive_unsigned("instance_id", self.instance_id, _UINT64_MAX)
        for name in (
            "num_layers",
            "experts_per_layer",
            "max_batch_tokens",
            "hidden_dim",
            "top_k",
        ):
            _require_positive_unsigned(name, getattr(self, name), _UINT32_MAX)
        if self.top_k > self.experts_per_layer:
            raise ValueError("top_k must not exceed experts_per_layer")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be FP16, BF16, or FP32")

    @property
    def activation_element_bytes(self) -> int:
        """Return the raw-byte width of one activation element."""

        return torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True, slots=True)
class BatchBufferConfig:
    """Describe fixed Tensor storage allocated for one execution slot."""

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
            raise ValueError("batch buffer device must be CPU or CUDA")
        object.__setattr__(self, "device", device)


class WorkerBatchBuffers(ABC):
    """Perform Transport-specific copies for one fixed execution slot.

    The methods run in the Worker's bounded execution thread. For CUDA, the
    Worker selects the current stream before calling them.
    """

    @property
    @abstractmethod
    def host_staging_bytes(self) -> int:
        """Return fixed Host bytes allocated by this Transport implementation."""

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
    def copy_output(
        self,
        partial_output: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        """Copy or expose a valid output view that this Transport can send."""

    @abstractmethod
    def close(self) -> None:
        """Release this slot's Transport-specific fixed resources."""


class ReceivedBatch(ABC):
    """Represent one admitted batch until computation and response finish."""

    @property
    @abstractmethod
    def trace_context(self) -> TraceContext | None:
        """Return optional Host-only tracing context captured by Transport."""

    @property
    @abstractmethod
    def batch(self) -> WorkerBatch:
        """Return the validated Host batch that must enter an execution slot."""

    @property
    @abstractmethod
    def monotonic_deadline(self) -> float:
        """Return the absolute end-to-end deadline observed by Transport."""

    @property
    @abstractmethod
    def cancelled(self) -> bool:
        """Return whether the caller no longer needs a response."""

    @property
    @abstractmethod
    def output_destination(self) -> torch.Tensor | None:
        """Return an optional Host Tensor owned by Transport for the response."""

    @abstractmethod
    def release_input(self) -> None:
        """Release received Tensor storage after copying it into an execution slot."""

    @abstractmethod
    async def complete(self, partial_output: torch.Tensor) -> None:
        """Finish communication of one successful Weighted partial output."""

    @abstractmethod
    async def reject(self, error: TransportError) -> None:
        """Finish the request with a structured computation rejection."""


class WorkerBatchReceiver(ABC):
    """Supply admitted batches and result communication to Worker execution."""

    @abstractmethod
    async def start(self) -> None:
        """Start the concrete Transport receiver."""

    @abstractmethod
    async def receive(self) -> ReceivedBatch:
        """Wait for and claim the next admitted batch."""

    @abstractmethod
    def create_batch_buffers(self, config: BatchBufferConfig) -> WorkerBatchBuffers:
        """Create Transport-specific fixed storage for one execution slot."""

    @abstractmethod
    async def begin_drain(
        self,
        experts: Iterable[tuple[int, int]],
        *,
        min_topology_version: int,
        stop_all: bool,
    ) -> None:
        """Reject new matching batches after Controller Topology cutover."""

    @abstractmethod
    async def clear_drains(self, experts: Iterable[tuple[int, int]]) -> None:
        """Allow newly assigned and ready experts after a later placement."""

    @abstractmethod
    async def wait_idle(
        self,
        experts: Iterable[tuple[int, int]] | None,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Wait for selected expert use, or all work when experts is `None`."""

    @abstractmethod
    async def close(self) -> None:
        """Stop admission and release Transport resources."""
