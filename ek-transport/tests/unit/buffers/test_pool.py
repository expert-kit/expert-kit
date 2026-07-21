"""Tests for bounded ordinary-Tensor output reuse."""

import asyncio

import pytest
import torch

from expertkit_transport.buffers import OutputPool
from expertkit_transport.errors import TransportError, TransportErrorCode


def pool(*, capacity: int = 1, clock=None) -> OutputPool:
    arguments = {
        "max_batch_tokens": 8,
        "hidden_dim": 4,
        "dtype": torch.float16,
        "device": "cpu",
        "capacity": capacity,
    }
    if clock is not None:
        arguments["clock"] = clock
    return OutputPool(**arguments)


def test_pool_preallocates_and_waits_without_growing() -> None:
    async def scenario() -> None:
        outputs = pool()
        second_waiting = asyncio.Event()
        second_acquired = asyncio.Event()

        async with outputs.lease(monotonic_deadline=float("inf")) as first:
            pointer = first.tensor.data_ptr()
            first.mark_consumed()

            async def acquire_second() -> None:
                second_waiting.set()
                async with outputs.lease(monotonic_deadline=float("inf")) as second:
                    assert second.tensor.data_ptr() == pointer
                    second_acquired.set()

            waiter = asyncio.create_task(acquire_second())
            await second_waiting.wait()
            assert not second_acquired.is_set()

        await waiter
        await outputs.close()

    asyncio.run(scenario())


def test_expired_deadline_does_not_consume_pool_capacity() -> None:
    async def scenario() -> None:
        outputs = pool(clock=lambda: 10.0)

        async with outputs.lease(monotonic_deadline=20.0):
            with pytest.raises(TransportError) as caught:
                async with outputs.lease(monotonic_deadline=10.0):
                    pass
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED

        async with outputs.lease(monotonic_deadline=20.0):
            pass
        await outputs.close()

    asyncio.run(scenario())


def test_cancelled_waiter_does_not_leak_a_tensor() -> None:
    async def scenario() -> None:
        outputs = pool()
        waiting = asyncio.Event()

        async with outputs.lease(monotonic_deadline=float("inf")):

            async def wait_for_tensor() -> None:
                waiting.set()
                async with outputs.lease(monotonic_deadline=float("inf")):
                    pass

            waiter = asyncio.create_task(wait_for_tensor())
            await waiting.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

        async with outputs.lease(monotonic_deadline=float("inf")):
            pass
        await outputs.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_batch_tokens": 0},
        {"hidden_dim": 0},
        {"capacity": 0},
        {"dtype": torch.int8},
    ],
)
def test_pool_rejects_invalid_tensor_configuration(arguments: dict[str, object]) -> None:
    configuration: dict[str, object] = {
        "max_batch_tokens": 8,
        "hidden_dim": 4,
        "dtype": torch.float16,
        "device": "cpu",
        "capacity": 1,
    }
    configuration.update(arguments)

    with pytest.raises(ValueError):
        OutputPool(**configuration)


def test_cancelled_close_caller_does_not_abandon_pool_cleanup() -> None:
    async def scenario() -> None:
        outputs = pool()
        acquired = asyncio.Event()
        release = asyncio.Event()

        async def hold_lease() -> None:
            async with outputs.lease(monotonic_deadline=float("inf")):
                acquired.set()
                await release.wait()

        holder = asyncio.create_task(hold_lease())
        await acquired.wait()
        close_caller = asyncio.create_task(outputs.close())
        await asyncio.sleep(0)
        close_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_caller

        release.set()
        await holder
        await outputs.close()

    asyncio.run(scenario())
