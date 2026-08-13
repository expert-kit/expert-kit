"""Tests for bounded ordinary-Tensor output reuse."""

import asyncio

import pytest
import torch

from expertkit_transport import buffers as buffers_module
from expertkit_transport.buffers import OutputLease, OutputPool
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


class _FailingEvent:
    def record(self, stream: object) -> None:
        del stream
        raise RuntimeError("injected event record failure")

    def synchronize(self) -> None:
        raise RuntimeError("injected event synchronize failure")


class _FailingStream:
    def wait_event(self, event: object) -> None:
        del event
        raise RuntimeError("injected stream wait failure")


class _FakeCudaTensor:
    device = torch.device("cuda:0")


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


def test_unsafe_transport_output_is_never_returned_to_the_pool() -> None:
    async def scenario() -> None:
        outputs = pool()
        error = TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=False,
            unsafe_output=True,
            diagnostic="CUDA output completion is unprovable",
        )

        with pytest.raises(TransportError) as caught:
            async with outputs.lease(monotonic_deadline=float("inf")):
                raise error
        assert caught.value is error
        assert not outputs._available  # type: ignore[attr-defined]
        assert len(outputs._quarantined) == 1  # type: ignore[attr-defined]
        with pytest.raises(TransportError, match="pool is unavailable"):
            async with outputs.lease(monotonic_deadline=float("inf")):
                pass
        await outputs.close()

    asyncio.run(scenario())


def test_ordinary_transport_error_still_returns_the_output_slot() -> None:
    async def scenario() -> None:
        outputs = pool()
        error = TransportError(
            TransportErrorCode.BUSY,
            retryable=True,
            diagnostic="ordinary admission rejection",
        )

        with pytest.raises(TransportError):
            async with outputs.lease(monotonic_deadline=float("inf")) as first:
                pointer = first.tensor.data_ptr()
                raise error
        async with outputs.lease(monotonic_deadline=float("inf")) as second:
            assert second.tensor.data_ptr() == pointer
        assert not outputs._quarantined  # type: ignore[attr-defined]
        await outputs.close()

    asyncio.run(scenario())


def test_reuse_event_record_failure_permanently_retains_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        outputs = pool()
        slot = outputs._available.pop()  # type: ignore[attr-defined]
        slot.tensor = _FakeCudaTensor()  # type: ignore[assignment]
        outputs._leased = 1  # type: ignore[attr-defined]
        lease = OutputLease(slot)
        owner_a = object()
        owner_b = object()
        lease.mark_consumed(ownership_graph=(owner_a,))
        lease.retain_consumption_owners(owner_b)
        monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: _FailingEvent())
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())

        with pytest.raises(TransportError, match="routing ownership graph") as caught:
            await outputs._return(lease)  # type: ignore[attr-defined]
        assert not caught.value.retryable
        assert caught.value.unsafe_tensor_ownership
        assert caught.value.unsafe_output
        assert any(  # type: ignore[attr-defined]
            candidate is slot for candidate in outputs._quarantined
        )
        assert any(
            candidate is slot
            for candidate in buffers_module._QUARANTINED_OUTPUT_SLOTS  # type: ignore[attr-defined]
        )
        assert not outputs._available  # type: ignore[attr-defined]
        retained = buffers_module._QUARANTINED_OUTPUT_GRAPHS[-1]  # type: ignore[attr-defined]
        assert any(candidate is owner_a for candidate in retained)
        assert any(candidate is owner_b for candidate in retained)
        assert any(candidate is slot for candidate in retained)
        await outputs.close()

    asyncio.run(scenario())


def test_reuse_event_wait_failure_never_restores_the_available_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        outputs = pool()
        slot = outputs._available[-1]  # type: ignore[attr-defined]
        slot.reuse_event = _FailingEvent()
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: _FailingStream())

        with pytest.raises(RuntimeError, match="stream wait failure"):
            await outputs._acquire(float("inf"))  # type: ignore[attr-defined]
        assert any(  # type: ignore[attr-defined]
            candidate is slot for candidate in outputs._quarantined
        )
        assert not outputs._available  # type: ignore[attr-defined]
        await outputs.close()

    asyncio.run(scenario())


def test_close_retains_a_slot_when_event_synchronize_fails() -> None:
    async def scenario() -> None:
        outputs = pool()
        slot = outputs._available[-1]  # type: ignore[attr-defined]
        slot.reuse_event = _FailingEvent()

        with pytest.raises(RuntimeError, match="event synchronize failure"):
            await outputs.close()
        assert any(  # type: ignore[attr-defined]
            candidate is slot for candidate in outputs._quarantined
        )
        assert any(
            candidate is slot
            for candidate in buffers_module._QUARANTINED_OUTPUT_SLOTS  # type: ignore[attr-defined]
        )

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


def test_repeated_cancellation_cannot_interrupt_lease_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        outputs = pool()
        return_started = asyncio.Event()
        allow_return = asyncio.Event()
        original_return = outputs._return  # type: ignore[attr-defined]

        async def delayed_return(
            lease: OutputLease,
            *,
            wait_for_completion: bool = False,
        ) -> None:
            del wait_for_completion
            return_started.set()
            await allow_return.wait()
            await original_return(lease)

        monkeypatch.setattr(outputs, "_return", delayed_return)

        async def acquire_and_exit() -> None:
            async with outputs.lease(monotonic_deadline=float("inf")):
                pass

        holder = asyncio.create_task(acquire_and_exit())
        await return_started.wait()
        holder.cancel()
        await asyncio.sleep(0)
        holder.cancel()
        await asyncio.sleep(0)
        allow_return.set()
        with pytest.raises(asyncio.CancelledError):
            await holder

        async with outputs.lease(monotonic_deadline=float("inf")):
            pass
        await outputs.close()

    asyncio.run(scenario())


def test_late_unsafe_error_takes_priority_over_exit_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        outputs = pool()
        return_started = asyncio.Event()
        allow_return = asyncio.Event()
        original_return = outputs._return  # type: ignore[attr-defined]
        unsafe = TransportError(
            TransportErrorCode.UNAVAILABLE,
            retryable=False,
            unsafe_tensor_ownership=True,
            unsafe_output=True,
            diagnostic="late CUDA staging ownership fatal",
        )

        async def delayed_return(
            lease: OutputLease,
            *,
            wait_for_completion: bool = False,
        ) -> None:
            del wait_for_completion
            return_started.set()
            await allow_return.wait()
            await original_return(lease)

        monkeypatch.setattr(outputs, "_return", delayed_return)

        async def fail_during_lease() -> None:
            async with outputs.lease(monotonic_deadline=float("inf")):
                raise unsafe

        holder = asyncio.create_task(fail_during_lease())
        await return_started.wait()
        holder.cancel()
        await asyncio.sleep(0)
        holder.cancel()
        allow_return.set()
        with pytest.raises(TransportError) as caught:
            await holder
        assert caught.value is unsafe
        assert len(outputs._quarantined) == 1  # type: ignore[attr-defined]
        await outputs.close()

    asyncio.run(scenario())
