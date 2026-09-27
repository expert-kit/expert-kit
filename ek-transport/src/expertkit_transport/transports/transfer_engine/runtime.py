"""Tensor compatibility facade over the independent Mooncake memory driver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch

from expertkit_transport.transports.transfer_engine.drivers.base import MemoryRegion, MemorySlice
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    MooncakeDriverConfig as TransferEngineRuntimeConfig,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    MooncakeMemoryTransport,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    _configured_backend_query as _configured_backend_query,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    _remote_invalidation_api as _remote_invalidation_api,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    _validate_configured_backend as _validate_configured_backend,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    _validate_native_capabilities as _validate_native_capabilities,
)


@runtime_checkable
class TransferEngineRuntimeProtocol(Protocol):
    """Injectable process-level contract used by clients and Worker receivers."""

    @property
    def device(self) -> torch.device: ...

    @property
    def session_id(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def generation(self) -> str: ...

    async def start(self) -> None: ...

    async def register_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None: ...

    async def unregister_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None: ...

    async def wait_event(
        self,
        event: torch.Event,
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def acquire_remote_writes(
        self,
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def batch_read(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def batch_write(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None: ...

    async def invalidate_remote_session(
        self,
        target_session: str,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None: ...

    async def close(self) -> None: ...

    def ensure_healthy(self) -> None: ...

    def quarantine(self, diagnostic: str) -> None: ...


class TransferEngineRuntime(MooncakeMemoryTransport):
    """Preserve the Tensor API used by existing TE expert client/receivers."""

    def _tensor_region(self, tensor: torch.Tensor) -> MemoryRegion:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Transfer Engine memory must be a Tensor")
        if tensor.device != self.device:
            raise ValueError("Transfer Engine Tensor device does not match the runtime")
        if not tensor.is_contiguous():
            raise ValueError("Transfer Engine Tensor must be contiguous")
        return MemoryRegion(
            tensor.data_ptr(), tensor.numel() * tensor.element_size(), str(tensor.device), tensor
        )

    async def register_tensor(
        self, tensor: torch.Tensor, *, monotonic_deadline: float = math.inf
    ) -> None:
        await self.register_region(
            self._tensor_region(tensor), monotonic_deadline=monotonic_deadline
        )

    async def unregister_tensor(
        self, tensor: torch.Tensor, *, monotonic_deadline: float = math.inf
    ) -> None:
        await self.unregister_region(
            self._tensor_region(tensor), monotonic_deadline=monotonic_deadline
        )

    def _tensor_slices(
        self, tensors: Sequence[torch.Tensor], addresses: Sequence[int], lengths: Sequence[int]
    ) -> tuple[MemorySlice, ...]:
        if not tensors or len(tensors) != len(addresses) or len(tensors) != len(lengths):
            raise ValueError("Transfer Engine batch vectors must have the same nonzero length")
        return tuple(
            MemorySlice(self._tensor_region(tensor), 0, address, length)
            for tensor, address, length in zip(tensors, addresses, lengths, strict=True)
        )

    async def batch_read(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None:
        await self.read(
            target_session,
            self._tensor_slices(local_tensors, remote_addresses, lengths),
            monotonic_deadline=monotonic_deadline,
        )

    async def batch_write(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None:
        await self.write(
            target_session,
            self._tensor_slices(local_tensors, remote_addresses, lengths),
            monotonic_deadline=monotonic_deadline,
        )

    async def batch_read_ready(
        self,
        target_session: str,
        local_tensors: Sequence[torch.Tensor],
        remote_addresses: Sequence[int],
        lengths: Sequence[int],
        *,
        monotonic_deadline: float,
    ) -> None:
        """Read all input regions and acquire them in one native submission."""
        await self._batch_transfer(
            "read",
            target_session,
            self._tensor_slices(local_tensors, remote_addresses, lengths),
            monotonic_deadline=monotonic_deadline,
            acquire_writes=True,
        )


__all__ = ["TransferEngineRuntime", "TransferEngineRuntimeConfig", "TransferEngineRuntimeProtocol"]
