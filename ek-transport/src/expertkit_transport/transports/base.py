"""Worker-side interface for taking admitted Transport batches."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class WorkerTransportRuntime(Protocol):
    """Own process-level resources shared by concrete Worker connections."""

    async def start(self) -> None:
        """Start shared communication resources without blocking on peer work."""

    async def close(self) -> None:
        """Release shared resources after every Worker connection is closed."""


class WorkerTransportRuntimeRegistry:
    """Resolve and own process-level runtimes used by Worker connections.

    Explicit entries support deployments that mix runtime-backed Transport
    types.  ``default_runtime`` preserves the original single-runtime client
    API for deployments that use at most one such Transport implementation.
    Concrete connections start their selected runtime lazily; the registry
    closes every distinct owned runtime exactly once.
    """

    def __init__(
        self,
        runtimes: Mapping[int, WorkerTransportRuntime] | None = None,
        *,
        default_runtime: WorkerTransportRuntime | None = None,
    ) -> None:
        resolved: dict[int, WorkerTransportRuntime] = {}
        for transport_type, runtime in (runtimes or {}).items():
            if (
                isinstance(transport_type, bool)
                or not isinstance(transport_type, int)
                or transport_type <= 0
            ):
                raise ValueError("Transport runtime type must be a positive integer")
            if not isinstance(runtime, WorkerTransportRuntime):
                raise TypeError("Transport runtimes must implement start() and close()")
            resolved[transport_type] = runtime
        if default_runtime is not None and not isinstance(
            default_runtime,
            WorkerTransportRuntime,
        ):
            raise TypeError("default Transport runtime must implement start() and close()")
        self._runtimes = resolved
        self._default_runtime = default_runtime
        self._close_task: asyncio.Task[None] | None = None

    def runtime_for(self, transport_type: int) -> WorkerTransportRuntime | None:
        """Return the explicitly registered runtime or the legacy default."""

        if isinstance(transport_type, bool) or not isinstance(transport_type, int):
            raise TypeError("transport_type must be an integer")
        return self._runtimes.get(transport_type, self._default_runtime)

    async def close(self) -> None:
        """Close every distinct runtime once, even when one close fails."""

        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_all(),
                name="worker-transport-runtime-registry-close",
            )
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        unique: dict[int, WorkerTransportRuntime] = {}
        for runtime in (*self._runtimes.values(), self._default_runtime):
            if runtime is not None:
                unique.setdefault(id(runtime), runtime)
        if unique:
            results = await asyncio.gather(
                *(runtime.close() for runtime in unique.values()),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result


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
    device: torch.device

    def __post_init__(self) -> None:
        for name in ("max_batch_tokens", "hidden_dim", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("dtype must be FP16, BF16, or FP32")
        if not isinstance(self.device, torch.device):
            raise TypeError("batch buffer device must be a torch.device")
        if self.device.type not in {"cpu", "cuda", "npu"}:
            raise ValueError("batch buffer device must be CPU, CUDA, or NPU")


class WorkerBatchBuffers(ABC):
    """Perform Transport-specific copies for one fixed execution slot.

    The methods run in the Worker's bounded execution thread. For accelerators,
    the Worker selects the current stream before calling them.
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

    def execution_views(
        self,
        batch: WorkerBatch,
        destination: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Borrow storage retained by the receiver until response completion.

        ``release_input()`` must not end the underlying arena lease. Transports
        without leased device buffers keep using the execution slot allocations.
        """
        return None


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

    async def reject_unsafe(self, error: TransportError) -> None:
        """Reject work whose device access cannot be proven terminal.

        Most Transports do not expose registered device storage, so their safe
        fallback is the ordinary rejection path.  A Transport that can retain
        remotely accessible storage must override this method and quarantine
        that storage before acknowledging the failure.
        """

        await self.reject(error)


class WorkerBatchReceiver(ABC):
    """Supply admitted batches and result communication to Worker execution."""

    @property
    def fixed_device_bytes(self) -> int:
        """Return device memory retained outside the execution-slot buffers."""

        return 0

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
