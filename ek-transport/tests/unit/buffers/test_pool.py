"""Tests for bounded output-buffer pool lifecycle."""

import asyncio

import pytest
import torch

from expertkit_transport.buffers import OutputPool
from expertkit_transport.contracts import (
    OutputBufferProvider,
    OutputSpec,
    PreparedOutput,
    TransportError,
    TransportErrorCode,
)


class FakeOutput(PreparedOutput):
    def __init__(self, spec: OutputSpec) -> None:
        self._tensor = torch.empty(
            (spec.max_batch_tokens, spec.hidden_dim),
            dtype=spec.dtype,
            device=spec.device,
        )

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor


class RecordingProvider(OutputBufferProvider):
    def __init__(self) -> None:
        self.prepared: list[PreparedOutput] = []
        self.before_receive_calls: list[PreparedOutput] = []
        self.after_consume_calls: list[PreparedOutput] = []
        self.released: list[PreparedOutput] = []

    def prepare(self, spec: OutputSpec) -> PreparedOutput:
        output = FakeOutput(spec)
        self.prepared.append(output)
        return output

    def validate(self, output: PreparedOutput, spec: OutputSpec) -> None:
        assert output.tensor.shape == (spec.max_batch_tokens, spec.hidden_dim)

    def before_receive(self, output: PreparedOutput) -> None:
        self.before_receive_calls.append(output)

    def after_consume(self, output: PreparedOutput) -> None:
        self.after_consume_calls.append(output)

    def release(self, output: PreparedOutput) -> None:
        self.released.append(output)


class RejectingProvider(RecordingProvider):
    def validate(self, output: PreparedOutput, spec: OutputSpec) -> None:
        raise ValueError("invalid prepared output")


class FailingCompletionProvider(RecordingProvider):
    def after_consume(self, output: PreparedOutput) -> None:
        raise RuntimeError("completion hook failed")


def test_pool_preallocates_and_waits_without_growing() -> None:
    async def scenario() -> None:
        provider = RecordingProvider()
        pool = OutputPool(provider, OutputSpec(8, 4, torch.float16, "cpu"), capacity=1)
        second_waiting = asyncio.Event()
        second_acquired = asyncio.Event()

        async with pool.lease(monotonic_deadline=float("inf")) as first:
            first.mark_consumed()

            async def acquire_second() -> None:
                second_waiting.set()
                async with pool.lease(monotonic_deadline=float("inf")):
                    second_acquired.set()

            waiter = asyncio.create_task(acquire_second())
            await second_waiting.wait()
            assert not second_acquired.is_set()

        await waiter
        assert len(provider.prepared) == 1
        assert provider.before_receive_calls == []
        assert provider.after_consume_calls == [provider.prepared[0]]
        await pool.close()
        assert provider.released == provider.prepared

    asyncio.run(scenario())


def test_expired_deadline_does_not_consume_pool_capacity() -> None:
    async def scenario() -> None:
        provider = RecordingProvider()
        pool = OutputPool(
            provider,
            OutputSpec(8, 4, torch.float16, "cpu"),
            capacity=1,
            clock=lambda: 10.0,
        )

        async with pool.lease(monotonic_deadline=20.0):
            with pytest.raises(TransportError) as caught:
                async with pool.lease(monotonic_deadline=10.0):
                    pass
            assert caught.value.code is TransportErrorCode.DEADLINE_EXCEEDED

        async with pool.lease(monotonic_deadline=20.0):
            pass
        await pool.close()

    asyncio.run(scenario())


def test_cancelled_waiter_does_not_leak_a_buffer() -> None:
    async def scenario() -> None:
        pool = OutputPool(
            RecordingProvider(),
            OutputSpec(8, 4, torch.float16, "cpu"),
            capacity=1,
        )
        waiting = asyncio.Event()

        async with pool.lease(monotonic_deadline=float("inf")):

            async def wait_for_buffer() -> None:
                waiting.set()
                async with pool.lease(monotonic_deadline=float("inf")):
                    pass

            waiter = asyncio.create_task(wait_for_buffer())
            await waiting.wait()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

        async with pool.lease(monotonic_deadline=float("inf")):
            pass
        await pool.close()

    asyncio.run(scenario())


def test_constructor_releases_outputs_after_validation_failure() -> None:
    provider = RejectingProvider()

    with pytest.raises(ValueError, match="invalid prepared output"):
        OutputPool(provider, OutputSpec(8, 4, torch.float16, "cpu"), capacity=2)

    assert provider.released == provider.prepared


def test_completion_hook_failure_closes_pool_before_reuse() -> None:
    async def scenario() -> None:
        provider = FailingCompletionProvider()
        pool = OutputPool(provider, OutputSpec(8, 4, torch.float16, "cpu"), capacity=1)

        with pytest.raises(RuntimeError, match="completion hook failed"):
            async with pool.lease(monotonic_deadline=float("inf")) as lease:
                lease.mark_consumed()

        with pytest.raises(TransportError) as caught:
            async with pool.lease(monotonic_deadline=float("inf")):
                pass
        assert caught.value.code is TransportErrorCode.UNAVAILABLE
        await pool.close()

    asyncio.run(scenario())


def test_cancelled_close_caller_does_not_abandon_pool_cleanup() -> None:
    async def scenario() -> None:
        provider = RecordingProvider()
        pool = OutputPool(provider, OutputSpec(8, 4, torch.float16, "cpu"), capacity=1)
        acquired = asyncio.Event()
        release = asyncio.Event()

        async def hold_lease() -> None:
            async with pool.lease(monotonic_deadline=float("inf")):
                acquired.set()
                await release.wait()

        holder = asyncio.create_task(hold_lease())
        await acquired.wait()
        close_caller = asyncio.create_task(pool.close())
        await asyncio.sleep(0)
        close_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_caller

        release.set()
        await holder
        await pool.close()
        assert provider.released == provider.prepared

    asyncio.run(scenario())
