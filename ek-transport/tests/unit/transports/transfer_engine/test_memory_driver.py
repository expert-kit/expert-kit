"""Byte-region bounds, ownership, and native completion compatibility."""

from __future__ import annotations

import asyncio
import ctypes
import math
import threading
import time
from types import SimpleNamespace

import pytest

from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.transfer_engine.drivers.base import (
    MemoryRegion,
    MemorySlice,
    MemoryTransport,
)
from expertkit_transport.transports.transfer_engine.drivers.mooncake import (
    _QUARANTINED_RUNTIMES,
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


class StagingEvent:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.recorded = threading.Event()

    def record(self) -> None:
        self.recorded.set()

    def query(self) -> bool:
        assert self.recorded.is_set(), "an unrecorded event cannot prove staging completion"
        return self.ready.is_set()


def staging_driver() -> MooncakeMemoryTransport:
    return MooncakeMemoryTransport(
        MooncakeDriverConfig("127.0.0.1", "P2PHANDSHAKE", "tcp", "cpu", max_workers=1),
        engine=ByteEngine(),
    )


def test_pending_staging_events_do_not_occupy_the_submission_thread() -> None:
    async def scenario() -> None:
        driver = staging_driver()
        events = [StagingEvent() for _ in range(5)]
        tasks = [
            asyncio.create_task(
                driver.run_device_operation(
                    event.record,
                    completion_event=event,
                    monotonic_deadline=math.inf,
                    subject="input staging",
                )
            )
            for event in events
        ]
        try:
            async with asyncio.timeout(2):
                while len(driver._staging_events) != len(events):
                    await asyncio.sleep(0)
            assert not any(task.done() for task in tasks)
            # Even with one submission thread, other native work can progress
            # while all five staging events remain incomplete.
            assert (
                await driver._run_native(
                    lambda: 17, monotonic_deadline=time.monotonic() + 1, subject="native probe"
                )
                == 17
            )
            assert len(driver._staging_events) == 5
            # A later stream may complete before earlier events. It must not
            # wait behind them in a FIFO completion queue.
            events[-1].ready.set()
            await asyncio.wait_for(asyncio.shield(tasks[-1]), 1)
            assert not any(task.done() for task in tasks[:-1])
            assert len(driver._staging_events) == 4
            for event in events:
                event.ready.set()
            await asyncio.gather(*tasks)
            driver.ensure_healthy()
        finally:
            for event in events:
                event.ready.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await driver.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("interrupt", ["cancel", "deadline", "close"])
def test_staging_keeps_owners_until_completion_after_interruption(interrupt: str) -> None:
    async def scenario() -> None:
        driver = staging_driver()
        event = StagingEvent()
        task = asyncio.create_task(
            driver.run_device_operation(
                event.record,
                completion_event=event,
                monotonic_deadline=time.monotonic() + 0.02 if interrupt == "deadline" else math.inf,
                subject="input staging",
            )
        )
        closing = None
        try:
            async with asyncio.timeout(2):
                while not driver._staging_events:
                    await asyncio.sleep(0)
            if interrupt == "cancel":
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            elif interrupt == "close":
                closing = asyncio.create_task(driver.close())
                await asyncio.sleep(0)
                with pytest.raises(TransportError, match="closing"):
                    await driver.run_device_operation(
                        lambda: None,
                        completion_event=StagingEvent(),
                        monotonic_deadline=math.inf,
                        subject="rejected staging",
                    )
            else:
                await asyncio.sleep(0.04)
            await asyncio.sleep(0)
            assert not task.done()
            if closing is not None:
                assert not closing.done()
            event.ready.set()
            if interrupt == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif interrupt == "deadline":
                with pytest.raises(TransportError) as error:
                    await task
                assert error.value.code is TransportErrorCode.DEADLINE_EXCEEDED
                driver.ensure_healthy()
            else:
                await task
                await closing
            assert not driver._staging_events
            assert not driver._operations
        finally:
            event.ready.set()
            await asyncio.gather(task, return_exceptions=True)
            await driver.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["query", "submit"])
def test_staging_failure_retains_registered_memory(failure: str) -> None:
    async def scenario() -> None:
        driver = staging_driver()
        owner = ctypes.create_string_buffer(16)
        region = MemoryRegion(ctypes.addressof(owner), len(owner), "cpu", owner)
        event = StagingEvent()
        if failure == "query":

            def broken_query() -> bool:
                raise RuntimeError("injected query failure")

            event.query = broken_query

        def submit() -> None:
            event.record()
            if failure == "submit":
                raise RuntimeError("injected submit failure after enqueue")

        await driver.register_region(region)
        with pytest.raises(RuntimeError, match="injected"):
            await driver.run_device_operation(
                submit,
                completion_event=event,
                monotonic_deadline=math.inf,
                subject="unsafe staging",
            )
        with pytest.raises(TransportError, match="quarantined"):
            driver.ensure_healthy()
        await driver.close()
        assert driver.registered_bytes == len(owner)
        assert driver._registrations[region.address][0] is owner
        # Only the injected CPU engine is safe to release after this test.
        _QUARANTINED_RUNTIMES.remove(driver)
        driver._executor.shutdown(wait=True)
        if driver._completion_executor is not None:
            driver._completion_executor.shutdown(wait=True)

    asyncio.run(scenario())


def test_cancelled_internal_staging_future_fails_closed_without_spinning() -> None:
    async def scenario() -> None:
        driver = staging_driver()
        owner = ctypes.create_string_buffer(16)
        region = MemoryRegion(ctypes.addressof(owner), len(owner), "cpu", owner)
        event = StagingEvent()
        await driver.register_region(region)
        caller = asyncio.create_task(
            driver.run_device_operation(
                event.record,
                completion_event=event,
                monotonic_deadline=math.inf,
                subject="input staging",
            )
        )
        try:
            async with asyncio.timeout(2):
                while not driver._staging_events:
                    await asyncio.sleep(0)
            internal = next(iter(driver._operations))
            internal.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 1)
            with pytest.raises(TransportError, match="quarantined"):
                driver.ensure_healthy()
            await driver.close()
            assert driver._registrations[region.address][0] is owner
        finally:
            event.ready.set()
            await asyncio.gather(caller, return_exceptions=True)
            _QUARANTINED_RUNTIMES.remove(driver)
            driver._executor.shutdown(wait=True)
            if driver._completion_executor is not None:
                driver._completion_executor.shutdown(wait=True)

    asyncio.run(scenario())
