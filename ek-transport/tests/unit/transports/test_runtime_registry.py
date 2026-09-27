"""Tests for process-level Transport runtime selection and ownership."""

import asyncio

import pytest

from expertkit_transport.transports.base import WorkerTransportRuntimeRegistry


class FakeRuntime:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.close_calls = 0
        self._fail_close = fail_close

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.close_calls += 1
        if self._fail_close:
            raise RuntimeError("close failed")


def test_explicit_runtime_overrides_legacy_default() -> None:
    default = FakeRuntime()
    transfer_engine = FakeRuntime()
    registry = WorkerTransportRuntimeRegistry({4: transfer_engine}, default_runtime=default)

    assert registry.runtime_for(4) is transfer_engine
    assert registry.runtime_for(3) is default


def test_registry_closes_duplicate_runtime_once_and_is_idempotent() -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        registry = WorkerTransportRuntimeRegistry(
            {3: runtime, 4: runtime},
            default_runtime=runtime,
        )

        await registry.close()
        await registry.close()

        assert runtime.close_calls == 1

    asyncio.run(scenario())


def test_registry_waits_for_all_runtime_closes_when_one_fails() -> None:
    async def scenario() -> None:
        failing = FakeRuntime(fail_close=True)
        healthy = FakeRuntime()
        registry = WorkerTransportRuntimeRegistry({3: failing, 4: healthy})

        with pytest.raises(RuntimeError, match="close failed"):
            await registry.close()

        assert failing.close_calls == 1
        assert healthy.close_calls == 1

    asyncio.run(scenario())


def test_registry_rejects_invalid_keys_and_runtime_contracts() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        WorkerTransportRuntimeRegistry({0: FakeRuntime()})
    with pytest.raises(TypeError, match=r"start.*close"):
        WorkerTransportRuntimeRegistry({3: object()})  # type: ignore[dict-item]
