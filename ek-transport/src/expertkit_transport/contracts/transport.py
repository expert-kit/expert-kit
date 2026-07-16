"""Frontend-side calling interface for one Worker Transport connection."""

from __future__ import annotations

from abc import ABC, abstractmethod

from expertkit_transport.contracts.batches import WorkerBatch
from expertkit_transport.contracts.buffers import OutputBufferProvider, PreparedOutput


class WorkerTransport(ABC):
    """Submit Worker batches to one remote Worker.

    Implementations are bound to an adapter-specific endpoint at construction.
    A successful `submit` fills `output.tensor[:batch.token_count]`. On every
    return or exception, including cancellation, the implementation must no
    longer access the batch inputs or output before allowing the caller to reuse
    them. Before overwriting a reused output, the adapter calls its provider's
    `before_receive` hook in the context that enqueues the actual transfer. The
    deadline is an absolute `time.monotonic()` value.
    """

    @property
    @abstractmethod
    def output_buffers(self) -> OutputBufferProvider:
        """Return hooks for preparing output buffers for this connection."""

    @abstractmethod
    async def start(self) -> None:
        """Create adapter resources on the current event loop."""

    @abstractmethod
    async def submit(
        self,
        batch: WorkerBatch,
        output: PreparedOutput,
        *,
        monotonic_deadline: float,
    ) -> None:
        """Fill one prepared output or raise a classified Transport error."""

    @abstractmethod
    async def close(self) -> None:
        """Stop submissions and release connection resources."""
