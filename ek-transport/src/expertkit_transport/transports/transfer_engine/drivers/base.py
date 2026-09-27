"""Owned byte regions and the model-independent memory transfer contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_UINT64_MAX = (1 << 64) - 1


def _positive_uint64(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _UINT64_MAX:
        raise ValueError(f"{name} must be a positive uint64")


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    """A contiguous allocation view with a strong reference to its owner."""

    address: int
    length: int
    device: str
    owner: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_uint64(self.address, "region address")
        _positive_uint64(self.length, "region length")
        if self.address + self.length > _UINT64_MAX + 1:
            raise ValueError("memory region address range overflows uint64")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("region device must not be empty")
        if self.owner is None:
            raise ValueError("memory region must retain an allocation owner")


@dataclass(frozen=True, slots=True)
class MemorySlice:
    """One READ/WRITE with validated local bounds and a remote byte address."""

    local: MemoryRegion
    local_offset: int
    remote_address: int
    length: int

    def __post_init__(self) -> None:
        if not isinstance(self.local, MemoryRegion):
            raise TypeError("local must be a MemoryRegion")
        offset = self.local_offset
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("local offset must be a nonnegative integer")
        _positive_uint64(self.length, "transfer length")
        _positive_uint64(self.remote_address, "remote address")
        if offset + self.length > self.local.length:
            raise ValueError("transfer exceeds a local Tensor view or memory region")
        if self.remote_address + self.length > _UINT64_MAX + 1:
            raise ValueError("remote transfer address range overflows uint64")


@runtime_checkable
class MemoryTransport(Protocol):
    """Transfer bytes without expert, framework, routing, or Controller types.

    Successful READ/WRITE returns only after native completion. Cancellation
    waits for submitted work; ambiguous native failures quarantine registrations.
    The adapter still establishes receive visibility before a device consumer.
    """

    async def start(self) -> None: ...

    async def register_region(
        self, region: MemoryRegion, *, monotonic_deadline: float = math.inf
    ) -> None: ...

    async def unregister_region(
        self, region: MemoryRegion, *, monotonic_deadline: float = math.inf
    ) -> None: ...

    async def read(
        self, target_session: str, slices: Sequence[MemorySlice], *, monotonic_deadline: float
    ) -> None: ...

    async def write(
        self, target_session: str, slices: Sequence[MemorySlice], *, monotonic_deadline: float
    ) -> None: ...

    async def acquire_remote_writes(self, *, monotonic_deadline: float) -> None: ...

    def ensure_healthy(self) -> None: ...

    def quarantine(self, diagnostic: str) -> None: ...

    async def close(self) -> None: ...
