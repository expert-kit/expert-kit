"""Byte-region bounds, ownership, and native completion compatibility."""

from __future__ import annotations

import asyncio
import ctypes
import math
from types import SimpleNamespace

import pytest

from expertkit_transport.transports.transfer_engine.drivers.base import (
    MemoryRegion,
    MemorySlice,
    MemoryTransport,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    MooncakeDriverConfig,
    MooncakeMemoryTransport,
    _validate_native_capabilities,
)


class ByteEngine:
    def initialize(self, *_args: object) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 19000

    def register_memory(self, *_args: object) -> int:
        return 0

    def unregister_memory(self, *_args: object) -> int:
        return 0

    def batch_transfer_sync_read(self, _peer, local, remote, sizes, _hint):
        for destination, source, length in zip(local, remote, sizes, strict=True):
            ctypes.memmove(destination, source, length)
        return 0

    def batch_transfer_sync_write(self, _peer, local, remote, sizes, _hint):
        return self.batch_transfer_sync_read(_peer, remote, local, sizes, _hint)


def test_byte_driver_transfers_without_model_or_tensor_arguments() -> None:
    async def scenario() -> None:
        source = ctypes.create_string_buffer(b"activation")
        target = ctypes.create_string_buffer(len(source))
        region = MemoryRegion(ctypes.addressof(target), len(target), "cpu", target)
        driver = MooncakeMemoryTransport(
            MooncakeDriverConfig("127.0.0.1", "P2PHANDSHAKE", "tcp", "cpu"),
            engine=ByteEngine(),
        )
        assert isinstance(driver, MemoryTransport)
        try:
            await driver.register_region(region)
            await driver.read(
                "peer:19001",
                [MemorySlice(region, 0, ctypes.addressof(source), len(source))],
                monotonic_deadline=math.inf,
            )
            assert target.raw == source.raw
            assert driver.registered_bytes == len(target)
            assert driver._registrations[region.address][0] is target
            await driver.unregister_region(region)
            assert driver.registered_bytes == 0
        finally:
            await driver.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "offset,length,remote", [(8, 9, 32), (-1, 1, 32), (0, 1, True), (0, 2, (1 << 64) - 1)]
)
def test_slice_rejects_invalid_ranges(offset: int, length: int, remote: int) -> None:
    with pytest.raises(ValueError):
        MemorySlice(MemoryRegion(128, 16, "cpu", object()), offset, remote, length)


def test_registration_budget_and_overlap_fail_before_native_registration() -> None:
    async def scenario() -> None:
        owner = ctypes.create_string_buffer(64)
        address = ctypes.addressof(owner)
        driver = MooncakeMemoryTransport(
            MooncakeDriverConfig(
                "127.0.0.1", "P2PHANDSHAKE", "tcp", "cpu", max_registered_bytes=32
            ),
            engine=ByteEngine(),
        )
        try:
            await driver.register_region(MemoryRegion(address, 16, "cpu", owner))
            with pytest.raises(ValueError, match="overlaps"):
                await driver.register_region(MemoryRegion(address + 8, 16, "cpu", owner))
            with pytest.raises(ValueError, match="budget exceeded"):
                await driver.register_region(MemoryRegion(address + 16, 32, "cpu", owner))
            assert driver.registered_bytes == 16
        finally:
            await driver.close()

    asyncio.run(scenario())


def test_ascend_requires_its_own_completion_and_visibility_capabilities() -> None:
    config = SimpleNamespace(protocol="ascend_direct", device=SimpleNamespace(type="npu"))
    with pytest.raises(RuntimeError, match="EK_ASCEND_SYNC_SUCCESS_COMPLETION"):
        _validate_native_capabilities(SimpleNamespace(EK_SAFE_TERMINAL_BATCH_SYNC=True), config)
    capabilities = {
        "EK_ASCEND_SYNC_SUCCESS_COMPLETION": True,
        "EK_HAS_ASCEND_ACQUIRE": True,
        "EK_FORCE_CONFIGURED_ASCEND_DIRECT_TRANSPORT": True,
        "EK_DRAINED_ASCEND_REMOTE_DESCRIPTOR_INVALIDATION": True,
    }
    _validate_native_capabilities(SimpleNamespace(**capabilities), config)
