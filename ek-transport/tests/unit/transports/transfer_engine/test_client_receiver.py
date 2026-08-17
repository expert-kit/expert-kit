"""Transfer Engine control/data behavior over an injectable CPU runtime."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import math
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass

import grpc
import pytest
import torch

from expertkit_transport.batches import WorkerBatch
from expertkit_transport.buffers import OutputPool
from expertkit_transport.errors import TransportError, TransportErrorCode
from expertkit_transport.transports.base import (
    BatchBufferConfig,
    ReceivedBatch,
    WorkerBatchBuffers,
    WorkerEndpointConfig,
)
from expertkit_transport.transports.transfer_engine import (
    TransferEngineWorkerBatchReceiver,
    TransferEngineWorkerTransport,
)
from expertkit_transport.transports.transfer_engine import client as client_module
from expertkit_transport.transports.transfer_engine.codec import (
    TransferExecutePull,
    TransferOpenSession,
    decode_admitted,
    decode_close_session,
    encode_admitted,
    encode_execute_pull_request,
    encode_open_session_request,
    encode_success,
)


@pytest.fixture(autouse=True)
def release_injected_process_quarantines_after_each_test() -> Iterator[None]:
    """Drop fake fail-stop graphs before Python interpreter finalization."""

    yield
    # Production intentionally retains these graphs until process exit.  The
    # injected test operations are already terminal and their fake channels
    # have been closed, so clear cross-test references while Python/grpc are
    # still fully initialized instead of relying on interpreter teardown.
    client_module._UNSAFE_STAGING_GRAPHS.clear()  # type: ignore[attr-defined]
    client_module._UNSAFE_CLOSE_GRAPHS.clear()  # type: ignore[attr-defined]
    gc.collect()


def endpoint_config() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(7, 4, 8, 4, 3, 2, torch.float32)


def worker_batch(*, offset: float = 0) -> WorkerBatch:
    return WorkerBatch(
        instance_id=7,
        layer_id=2,
        topology_version=11,
        hidden_states=torch.tensor(
            [[1 + offset, 2 + offset, 3 + offset], [4, 5, 6], [7, 8, 9]],
            dtype=torch.float32,
        ),
        token_indices=torch.tensor([2, 0], dtype=torch.int64),
        expert_ids=torch.tensor([[1, -1], [0, 3]], dtype=torch.int32),
        routing_weights=torch.tensor([[0.25, 0], [0.5, 0.5]], dtype=torch.float32),
        distinct_expert_ids=(0, 1, 3),
    )


class _FakeLink:
    def __init__(self) -> None:
        self.regions: dict[str, list[tuple[int, int, torch.Tensor]]] = {}
        self.operations: list[tuple[str, str, int]] = []
        self.write_started = asyncio.Event()
        self.allow_write = asyncio.Event()
        self.block_writes = False

    def register(self, session: str, tensor: torch.Tensor) -> None:
        self.regions.setdefault(session, []).append(
            (tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor)
        )

    def unregister(self, session: str, tensor: torch.Tensor) -> None:
        regions = self.regions[session]
        regions[:] = [entry for entry in regions if entry[2] is not tensor]

    def require_remote(self, session: str, address: int, length: int) -> None:
        assert any(
            base <= address and address + length <= base + capacity
            for base, capacity, _tensor in self.regions[session]
        )


class _FakeRuntime:
    def __init__(
        self,
        link: _FakeLink,
        session_id: str,
        *,
        backend: str = "tcp",
        generation: str = "runtime-generation-a",
    ) -> None:
        self._link = link
        self._session_id = session_id
        self._backend = backend
        self._generation = generation
        self._closed = False
        self.quarantine_reason: str | None = None
        self.invalidated_sessions: list[str] = []
        self.acquire_count = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def generation(self) -> str:
        return self._generation

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("fake Transfer Engine runtime is closed")

    async def register_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        del monotonic_deadline
        self._link.register(self._session_id, tensor)

    async def unregister_tensor(
        self,
        tensor: torch.Tensor,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        del monotonic_deadline
        self._link.unregister(self._session_id, tensor)

    async def wait_event(
        self,
        event: torch.cuda.Event,
        *,
        monotonic_deadline: float,
    ) -> None:
        del event, monotonic_deadline

    async def acquire_remote_writes(
        self,
        *,
        monotonic_deadline: float,
    ) -> None:
        assert monotonic_deadline > time.monotonic() or math.isinf(monotonic_deadline)
        self.acquire_count += 1

    async def batch_read(
        self,
        target_session: str,
        local_tensors: tuple[torch.Tensor, ...],
        remote_addresses: tuple[int, ...],
        lengths: tuple[int, ...],
        *,
        monotonic_deadline: float,
    ) -> None:
        assert monotonic_deadline > time.monotonic()
        self._link.operations.append(("read", target_session, len(lengths)))
        for local, remote, length in zip(local_tensors, remote_addresses, lengths, strict=True):
            self._link.require_remote(target_session, remote, length)
            ctypes.memmove(local.data_ptr(), remote, length)

    async def batch_write(
        self,
        target_session: str,
        local_tensors: tuple[torch.Tensor, ...],
        remote_addresses: tuple[int, ...],
        lengths: tuple[int, ...],
        *,
        monotonic_deadline: float,
    ) -> None:
        assert math.isinf(monotonic_deadline)
        self._link.operations.append(("write", target_session, len(lengths)))
        self._link.write_started.set()
        if self._link.block_writes:
            await self._link.allow_write.wait()
        for local, remote, length in zip(local_tensors, remote_addresses, lengths, strict=True):
            self._link.require_remote(target_session, remote, length)
            ctypes.memmove(remote, local.data_ptr(), length)

    async def invalidate_remote_session(
        self,
        target_session: str,
        *,
        monotonic_deadline: float = math.inf,
    ) -> None:
        assert math.isinf(monotonic_deadline)
        self.invalidated_sessions.append(target_session)

    async def close(self) -> None:
        self._closed = True

    def ensure_healthy(self) -> None:
        if self.quarantine_reason is not None:
            raise TransportError(
                TransportErrorCode.UNAVAILABLE,
                retryable=False,
                diagnostic="fake Transfer Engine runtime is quarantined",
            )

    def quarantine(self, diagnostic: str) -> None:
        self.quarantine_reason = diagnostic


class _InjectedCudaEvent:
    def __init__(self, *, fail_record: int | None = None) -> None:
        self.record_count = 0
        self.fail_record = fail_record

    def record(self, stream: object) -> None:
        del stream
        self.record_count += 1
        if self.record_count == self.fail_record:
            raise RuntimeError("injected CUDA event record failure")

    def synchronize(self) -> None:
        return None


@dataclass(slots=True)
class _RunningPair:
    receiver: TransferEngineWorkerBatchReceiver
    transport: TransferEngineWorkerTransport
    buffers: WorkerBatchBuffers
    link: _FakeLink
    client_runtime: _FakeRuntime
    worker_runtime: _FakeRuntime

    async def close(self) -> None:
        try:
            await self.transport.close()
        finally:
            try:
                await self.receiver.close()
            finally:
                self.buffers.close()
                await self.client_runtime.close()
                await self.worker_runtime.close()


async def start_pair(
    *,
    max_active_batches: int = 1,
    max_pending_batches: int = 1,
    max_in_flight: int = 2,
    session_close_grace_secs: float = 0.2,
    backend: str = "tcp",
) -> _RunningPair:
    link = _FakeLink()
    client_runtime = _FakeRuntime(link, "client:19001", backend=backend)
    worker_runtime = _FakeRuntime(link, "worker:19002", backend=backend)
    config = endpoint_config()
    receiver = TransferEngineWorkerBatchReceiver(
        "127.0.0.1:0",
        config,
        runtime=worker_runtime,
        worker_start_id="worker-start-a",
        max_active_batches=max_active_batches,
        max_pending_batches=max_pending_batches,
        session_close_grace_secs=session_close_grace_secs,
        owns_runtime=False,
    )
    buffers = receiver.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
    await receiver.start()
    transport = TransferEngineWorkerTransport(
        f"127.0.0.1:{receiver.bound_port}",
        config,
        max_in_flight=max_in_flight,
        device="cpu",
        runtime=client_runtime,
        expected_worker_start_id="worker-start-a",
    )
    await transport.start()
    return _RunningPair(
        receiver,
        transport,
        buffers,
        link,
        client_runtime,
        worker_runtime,
    )


async def complete(received: ReceivedBatch, *, multiplier: float = 2) -> None:
    partial_output = received.batch.hidden_states * multiplier
    destination = received.output_destination
    assert destination is not None
    destination.copy_(partial_output)
    received.release_input()
    await received.complete(destination)


def test_worker_pulls_three_inputs_and_writes_one_output() -> None:
    async def scenario() -> None:
        pair = await start_pair()
        batch = worker_batch()
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            submission = asyncio.create_task(
                pair.transport.execute(
                    batch,
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await pair.receiver.receive()
            batch.hidden_states.fill_(99)
            assert received.batch.hidden_states.tolist() == [[7, 8, 9], [1, 2, 3]]
            await complete(received)
            await submission

            torch.testing.assert_close(
                output,
                torch.tensor([[14, 16, 18], [2, 4, 6]], dtype=torch.float32),
            )
            assert pair.link.operations == [
                ("read", "client:19001", 3),
                ("write", "client:19001", 1),
            ]
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_rdma_backend_negotiates_and_uses_two_phase_invalidation() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1, backend="rdma")
        output = torch.empty((2, 3), dtype=torch.float32)
        try:
            submission = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(),
                    output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await pair.receiver.receive()
            assert received.batch.distinct_expert_ids == (0, 1, 3)
            await complete(received)
            await submission

            await pair.transport.close()
            assert pair.worker_runtime.invalidated_sessions == ["client:19001"]
            assert not pair.receiver._sessions  # type: ignore[attr-defined]
            assert pair.client_runtime.quarantine_reason is None
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_unsafe_rejection_quarantines_worker_slot_and_registered_arena() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_active_batches=1, max_pending_batches=1)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            received = await pair.receiver.receive()
            worker_slot = received.slot  # type: ignore[attr-defined]
            await received.reject_unsafe(
                TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=True,
                    diagnostic="CUDA completion is ambiguous",
                )
            )
            with pytest.raises(TransportError) as captured:
                await submission

            assert captured.value.code is TransportErrorCode.UNAVAILABLE
            assert captured.value.retryable is False
            assert pair.worker_runtime.quarantine_reason is not None
            assert worker_slot.in_use
            assert worker_slot in pair.receiver._quarantined_slots  # type: ignore[attr-defined]

            assert pair.receiver._registered  # type: ignore[attr-defined]
            assert pair.link.regions[pair.worker_runtime.session_id]
        finally:
            if not submission.done():
                submission.cancel()
            await asyncio.gather(submission, return_exceptions=True)
            server = pair.receiver._server  # type: ignore[attr-defined]
            if server is not None:
                await server.stop(0)
            channel = pair.transport._channel  # type: ignore[attr-defined]
            if channel is not None:
                await channel.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["read", "write"])
def test_unexpected_native_batch_exception_retains_both_registered_arenas(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_active_batches=1, max_pending_batches=1, max_in_flight=1)
        output = torch.empty((2, 3), dtype=torch.float32)
        receive_task: asyncio.Task[ReceivedBatch] | None = None
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                output,
                monotonic_deadline=time.monotonic() + 5,
            )
        )

        async def fail_native(*args: object, **kwargs: object) -> None:
            del args, kwargs
            pair.worker_runtime.quarantine(
                f"injected native {operation} exception before terminal DMA state"
            )
            raise RuntimeError(f"injected native {operation} exception")

        monkeypatch.setattr(pair.worker_runtime, f"batch_{operation}", fail_native)
        try:
            if operation == "read":
                # The receiver loop must consume the failed admitted item so
                # queue/session ownership reaches its fail-closed terminal.
                receive_task = asyncio.create_task(pair.receiver.receive())
            else:
                received = await pair.receiver.receive()
                destination = received.output_destination
                assert destination is not None
                destination.copy_(received.batch.hidden_states * 2)
                received.release_input()
                with pytest.raises(RuntimeError, match="native write exception"):
                    await received.complete(destination)

            with pytest.raises(TransportError):
                await submission

            for _ in range(100):
                if pair.receiver._quarantined_slots:  # type: ignore[attr-defined]
                    break
                await asyncio.sleep(0.001)
            worker_slot = pair.receiver._arena.slots[0]  # type: ignore[attr-defined]
            client_slot = pair.transport._arena.slots[0]  # type: ignore[attr-defined]
            assert pair.worker_runtime.quarantine_reason is not None
            assert pair.client_runtime.quarantine_reason is not None
            assert worker_slot in pair.receiver._quarantined_slots  # type: ignore[attr-defined]
            assert client_slot in pair.transport._quarantined_slots  # type: ignore[attr-defined]
            assert worker_slot.in_use
            assert client_slot.in_use
            assert pair.receiver._registered  # type: ignore[attr-defined]
            assert pair.link.regions[pair.worker_runtime.session_id]
            assert pair.link.regions[pair.client_runtime.session_id]
        finally:
            for task in (submission, receive_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (submission, receive_task) if task is not None),
                return_exceptions=True,
            )
            server = pair.receiver._server  # type: ignore[attr-defined]
            if server is not None:
                await server.stop(0)
            channel = pair.transport._channel  # type: ignore[attr-defined]
            if channel is not None:
                await channel.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_repeated_cancel_during_terminal_release_still_publishes_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        original_finish = pair.receiver._queue.finish  # type: ignore[attr-defined]

        async def blocked_finish(item: object) -> None:
            release_started.set()
            await allow_release.wait()
            await original_finish(item)

        monkeypatch.setattr(pair.receiver._queue, "finish", blocked_finish)  # type: ignore[attr-defined]
        output = torch.empty((2, 3), dtype=torch.float32)
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                output,
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        completion: asyncio.Task[None] | None = None
        try:
            received = await pair.receiver.receive()
            destination = received.output_destination
            assert destination is not None
            destination.copy_(received.batch.hidden_states * 2)
            received.release_input()
            completion = asyncio.create_task(received.complete(destination))
            await asyncio.wait_for(release_started.wait(), 2)

            completion.cancel()
            completion.cancel()
            await asyncio.sleep(0)
            assert not completion.done()
            allow_release.set()
            await completion
            await submission

            assert pair.receiver.active_count == 0
            session = next(iter(pair.receiver._sessions.values()))  # type: ignore[attr-defined]
            assert session.idle
            assert all(
                not slot.in_use
                for slot in pair.receiver._arena.slots  # type: ignore[attr-defined]
            )
        finally:
            allow_release.set()
            for task in (submission, completion):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (submission, completion) if task is not None),
                return_exceptions=True,
            )
            await pair.close()

    asyncio.run(scenario())


def test_slot_is_not_reused_until_output_write_is_terminal() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        pair.link.block_writes = True
        first_output = torch.empty((2, 3), dtype=torch.float32)
        second_output = torch.empty((2, 3), dtype=torch.float32)
        first = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                first_output,
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        second: asyncio.Task[None] | None = None
        completion: asyncio.Task[None] | None = None
        try:
            received = await pair.receiver.receive()
            destination = received.output_destination
            assert destination is not None
            destination.copy_(received.batch.hidden_states * 2)
            received.release_input()
            completion = asyncio.create_task(received.complete(destination))
            await asyncio.wait_for(pair.link.write_started.wait(), 2)

            second = asyncio.create_task(
                pair.transport.execute(
                    worker_batch(offset=10),
                    second_output,
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            await asyncio.sleep(0.05)
            assert not first.done()
            assert not second.done()

            pair.link.block_writes = False
            pair.link.allow_write.set()
            await completion
            await first
            second_received = await pair.receiver.receive()
            await complete(second_received)
            await second
        finally:
            pair.link.block_writes = False
            pair.link.allow_write.set()
            for task in (first, second, completion):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first, second, completion) if task is not None),
                return_exceptions=True,
            )
            await pair.close()

    asyncio.run(scenario())


def test_open_session_rejects_a_stale_worker_start_id() -> None:
    async def scenario() -> None:
        link = _FakeLink()
        worker_runtime = _FakeRuntime(link, "worker:19002")
        client_runtime = _FakeRuntime(link, "client:19001")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="current-start",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()
        transport = TransferEngineWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=client_runtime,
            expected_worker_start_id="stale-start",
        )
        try:
            with pytest.raises(Exception, match=r"start ID|INVALID_ARGUMENT"):
                await transport.start()
        finally:
            await transport.close()
            await receiver.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_ambiguous_post_admission_failure_quarantines_the_frontend_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)

        async def fail_after_admission(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise client_module._AmbiguousRemoteState(  # type: ignore[attr-defined]
                TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=True,
                    diagnostic="control stream disappeared",
                )
            )

        monkeypatch.setattr(pair.transport, "_execute_slot", fail_after_admission)
        try:
            with pytest.raises(TransportError, match="control stream disappeared"):
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert pair.client_runtime.quarantine_reason is not None
            assert pair.transport._arena.slots[0].in_use  # type: ignore[attr-defined]
            with pytest.raises(TransportError, match="process must restart"):
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_missing_admission_frame_quarantines_the_frontend_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingAdmissionCall:
        async def read(self) -> object:
            return grpc.aio.EOF

        def cancel(self) -> bool:
            return True

    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        captured_timeout: float | None = None

        def missing_admission(*args: object, **kwargs: object) -> _MissingAdmissionCall:
            nonlocal captured_timeout
            del args
            timeout = kwargs.get("timeout")
            assert isinstance(timeout, float)
            captured_timeout = timeout
            return _MissingAdmissionCall()

        monkeypatch.setattr(pair.transport, "_execute", missing_admission)
        try:
            with pytest.raises(TransportError, match="ended before admission"):
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert captured_timeout is not None and math.isfinite(captured_timeout)
            assert pair.client_runtime.quarantine_reason is not None
            assert pair.transport._arena.slots[0].in_use  # type: ignore[attr-defined]
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_first_frame_success_is_ambiguous_and_quarantines_the_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SuccessBeforeAdmissionCall:
        async def read(self) -> bytes:
            return encode_success(1)

        def cancel(self) -> bool:
            return True

    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        monkeypatch.setattr(
            pair.transport,
            "_execute",
            lambda *args, **kwargs: _SuccessBeforeAdmissionCall(),
        )
        try:
            with pytest.raises(TransportError, match="version or kind is invalid") as captured:
                await pair.transport.execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert captured.value.retryable is False
            assert pair.client_runtime.quarantine_reason is not None
            assert pair.transport._arena.slots[0].in_use  # type: ignore[attr-defined]
        finally:
            await pair.close()

    asyncio.run(scenario())


def test_each_close_invalidates_a_drained_shared_remote_segment() -> None:
    async def scenario() -> None:
        link = _FakeLink()
        client_runtime = _FakeRuntime(link, "client:19001")
        worker_runtime = _FakeRuntime(link, "worker:19002")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()
        transports = [
            TransferEngineWorkerTransport(
                f"127.0.0.1:{receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=client_runtime,
                expected_worker_start_id="worker-start-a",
            )
            for _ in range(2)
        ]
        try:
            for transport in transports:
                await transport.start()
            await transports[0].close()
            assert worker_runtime.invalidated_sessions == ["client:19001"]
            await transports[1].close()
            assert worker_runtime.invalidated_sessions == [
                "client:19001",
                "client:19001",
            ]
        finally:
            for transport in transports:
                await transport.close()
            await receiver.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_two_phase_close_gates_sibling_between_deregister_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        link = _FakeLink()
        client_runtime = _FakeRuntime(link, "client:19001")
        worker_runtime = _FakeRuntime(link, "worker:19002")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        buffers = receiver.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
        await receiver.start()
        transports = [
            TransferEngineWorkerTransport(
                f"127.0.0.1:{receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=client_runtime,
                expected_worker_start_id="worker-start-a",
            )
            for _ in range(2)
        ]
        commit_started = asyncio.Event()
        allow_commit = asyncio.Event()
        original_invalidate = worker_runtime.invalidate_remote_session

        async def blocked_commit(
            target_session: str,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            commit_started.set()
            await allow_commit.wait()
            await original_invalidate(
                target_session,
                monotonic_deadline=monotonic_deadline,
            )

        monkeypatch.setattr(worker_runtime, "invalidate_remote_session", blocked_commit)
        closing: asyncio.Task[None] | None = None
        try:
            for transport in transports:
                await transport.start()
            closing_arena = transports[0]._arena.slab  # type: ignore[attr-defined]
            closing = asyncio.create_task(transports[0].close())
            await commit_started.wait()

            assert all(
                tensor is not closing_arena
                for _base, _capacity, tensor in link.regions["client:19001"]
            )
            assert receiver._target_close_gates == {  # type: ignore[attr-defined]
                "client:19001": {transports[0]._client_epoch}  # type: ignore[attr-defined]
            }
            with pytest.raises(TransportError) as gated:
                await transports[1].execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert gated.value.code is TransportErrorCode.UNAVAILABLE
            assert gated.value.retryable is True
            assert worker_runtime.invalidated_sessions == []

            allow_commit.set()
            await closing
            assert worker_runtime.invalidated_sessions == ["client:19001"]
            assert not receiver._target_close_gates  # type: ignore[attr-defined]

            submission = asyncio.create_task(
                transports[1].execute(
                    worker_batch(offset=10),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await receiver.receive()
            await complete(received)
            await submission
        finally:
            allow_commit.set()
            if closing is not None and not closing.done():
                closing.cancel()
            await asyncio.gather(*(transport.close() for transport in transports))
            await receiver.close()
            buffers.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_concurrent_prepared_epochs_hold_gate_until_every_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        link = _FakeLink()
        client_runtime = _FakeRuntime(link, "client:19001")
        worker_runtime = _FakeRuntime(link, "worker:19002")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        buffers = receiver.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
        await receiver.start()
        transports = [
            TransferEngineWorkerTransport(
                f"127.0.0.1:{receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=client_runtime,
                expected_worker_start_id="worker-start-a",
            )
            for _ in range(3)
        ]
        original_unregister = client_runtime.unregister_tensor
        both_deregistered = asyncio.Event()
        allow_commits = asyncio.Event()
        deregister_count = 0

        async def block_after_deregister(
            tensor: torch.Tensor,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            nonlocal deregister_count
            await original_unregister(tensor, monotonic_deadline=monotonic_deadline)
            deregister_count += 1
            if deregister_count == 2:
                both_deregistered.set()
            await allow_commits.wait()

        monkeypatch.setattr(client_runtime, "unregister_tensor", block_after_deregister)
        closers: list[asyncio.Task[None]] = []
        try:
            for transport in transports:
                await transport.start()
            closers = [asyncio.create_task(transports[index].close()) for index in range(2)]
            await both_deregistered.wait()
            assert receiver._target_close_gates == {  # type: ignore[attr-defined]
                "client:19001": {
                    transports[0]._client_epoch,  # type: ignore[attr-defined]
                    transports[1]._client_epoch,  # type: ignore[attr-defined]
                }
            }
            assert worker_runtime.invalidated_sessions == []

            with pytest.raises(TransportError) as gated:
                await transports[2].execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert gated.value.code is TransportErrorCode.UNAVAILABLE
            assert gated.value.retryable is True

            allow_commits.set()
            await asyncio.gather(*closers)
            assert worker_runtime.invalidated_sessions == [
                "client:19001",
                "client:19001",
            ]
            assert not receiver._target_close_gates  # type: ignore[attr-defined]

            submission = asyncio.create_task(
                transports[2].execute(
                    worker_batch(offset=10),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await receiver.receive()
            await complete(received)
            await submission
        finally:
            allow_commits.set()
            await asyncio.gather(*closers, return_exceptions=True)
            await asyncio.gather(*(transport.close() for transport in transports))
            await receiver.close()
            buffers.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_close_waits_past_start_timeout_for_active_sibling_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        link = _FakeLink()
        client_runtime = _FakeRuntime(link, "client:19001")
        worker_runtime = _FakeRuntime(link, "worker:19002")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        buffers = receiver.create_batch_buffers(BatchBufferConfig(4, 3, 2, torch.float32, "cpu"))
        await receiver.start()
        transports = [
            TransferEngineWorkerTransport(
                f"127.0.0.1:{receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=client_runtime,
                expected_worker_start_id="worker-start-a",
            )
            for _ in range(2)
        ]
        submission: asyncio.Task[None] | None = None
        closing: asyncio.Task[None] | None = None
        try:
            for transport in transports:
                await transport.start()
            monkeypatch.setattr(client_module, "_START_TIMEOUT_SECONDS", 0.01)
            submission = asyncio.create_task(
                transports[1].execute(
                    worker_batch(),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            received = await receiver.receive()
            closing = asyncio.create_task(transports[0].close())
            await asyncio.sleep(0.05)

            assert not closing.done()
            assert worker_runtime.invalidated_sessions == []
            assert link.regions["client:19001"]

            await complete(received)
            await submission
            await closing
            assert worker_runtime.invalidated_sessions == ["client:19001"]

            # The other session survives the target-wide cache invalidation and
            # can lazily reopen the mapping on its next request.
            next_submission = asyncio.create_task(
                transports[1].execute(
                    worker_batch(offset=10),
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            )
            next_received = await receiver.receive()
            await complete(next_received)
            await next_submission
        finally:
            for task in (submission, closing):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (submission, closing) if task is not None),
                return_exceptions=True,
            )
            for transport in transports:
                await transport.close()
            await receiver.close()
            buffers.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_lost_open_ack_quarantines_and_retains_the_registered_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        link = _FakeLink()
        client_runtime = _FakeRuntime(link, "client:19001")
        worker_runtime = _FakeRuntime(link, "worker:19002")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            session_close_grace_secs=0.01,
            owns_runtime=False,
        )
        original_open = receiver._open_session  # type: ignore[attr-defined]

        async def commit_then_lose_ack(
            payload: bytes,
            context: grpc.aio.ServicerContext,
        ) -> bytes:
            await original_open(payload, context)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "lost OpenSession ack")
            raise AssertionError("context.abort must terminate the handler")

        monkeypatch.setattr(receiver, "_open_session", commit_then_lose_ack)
        await receiver.start()
        transport = TransferEngineWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=client_runtime,
            expected_worker_start_id="worker-start-a",
        )
        try:
            with pytest.raises(TransportError, match="may have committed") as captured:
                await transport.start()
            assert captured.value.retryable is False
            assert client_runtime.quarantine_reason is not None
            assert len(receiver._sessions) == 1  # type: ignore[attr-defined]
            assert link.regions["client:19001"]

            await transport.close()
            assert link.regions["client:19001"]
        finally:
            await receiver.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_closed_session_epoch_cannot_be_reopened() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        request = TransferOpenSession(
            client_epoch=pair.transport._client_epoch,  # type: ignore[attr-defined]
            client_session_id=pair.client_runtime.session_id,
            client_runtime_generation=pair.client_runtime.generation,
            expected_worker_start_id="worker-start-a",
            backend=pair.client_runtime.backend,
            arena=pair.transport._arena.descriptor,  # type: ignore[attr-defined]
        )
        replay = encode_open_session_request(request, endpoint_config())
        channel: grpc.aio.Channel | None = None
        try:
            await pair.transport.close()
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{pair.receiver.bound_port}")
            open_session = channel.unary_unary(
                "/ek.worker.v2.TransferEngineComputationService/OpenSession",
                request_serializer=lambda payload: payload,
                response_deserializer=lambda payload: payload,
            )
            with pytest.raises(grpc.aio.AioRpcError) as captured:
                await open_session(replay, timeout=2, wait_for_ready=False)
            assert captured.value.code() is grpc.StatusCode.INVALID_ARGUMENT
            assert "closed" in captured.value.details()
        finally:
            if channel is not None:
                await channel.close()
            await pair.close()

    asyncio.run(scenario())


def test_open_session_rejects_a_backend_mismatch() -> None:
    async def scenario() -> None:
        link = _FakeLink()
        worker_runtime = _FakeRuntime(link, "worker:19002", backend="tcp")
        client_runtime = _FakeRuntime(link, "client:19001", backend="rdma")
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()
        transport = TransferEngineWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=client_runtime,
            expected_worker_start_id="worker-start-a",
        )
        try:
            with pytest.raises(TransportError, match="INVALID_ARGUMENT"):
                await transport.start()
        finally:
            await transport.close()
            await receiver.close()
            await client_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_nvlink_same_endpoint_allows_new_generation_only_after_close() -> None:
    async def scenario() -> None:
        link = _FakeLink()
        worker_runtime = _FakeRuntime(
            link,
            "worker:19002",
            backend="nvlink_intra",
        )
        original_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend="nvlink_intra",
            generation="frontend-generation-a",
        )
        restarted_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend="nvlink_intra",
            generation="frontend-generation-b",
        )
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()
        original = TransferEngineWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=original_runtime,
            expected_worker_start_id="worker-start-a",
        )
        restarted = TransferEngineWorkerTransport(
            f"127.0.0.1:{receiver.bound_port}",
            endpoint_config(),
            max_in_flight=1,
            device="cpu",
            runtime=restarted_runtime,
            expected_worker_start_id="worker-start-a",
        )
        try:
            await original.start()
            with pytest.raises(TransportError, match="INVALID_ARGUMENT"):
                await restarted.start()
            await original.close()
            await restarted.start()
        finally:
            await restarted.close()
            await original.close()
            await receiver.close()
            await original_runtime.close()
            await restarted_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_rdma_closed_endpoint_keeps_generation_tombstone() -> None:
    async def scenario() -> None:
        link = _FakeLink()
        worker_runtime = _FakeRuntime(link, "worker:19002", backend="rdma")
        original_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend="rdma",
            generation="frontend-generation-a",
        )
        same_generation_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend="rdma",
            generation="frontend-generation-a",
        )
        restarted_runtime = _FakeRuntime(
            link,
            "client:19001",
            backend="rdma",
            generation="frontend-generation-b",
        )
        receiver = TransferEngineWorkerBatchReceiver(
            "127.0.0.1:0",
            endpoint_config(),
            runtime=worker_runtime,
            worker_start_id="worker-start-a",
            max_active_batches=1,
            max_pending_batches=1,
            owns_runtime=False,
        )
        await receiver.start()

        def transport(runtime: _FakeRuntime) -> TransferEngineWorkerTransport:
            return TransferEngineWorkerTransport(
                f"127.0.0.1:{receiver.bound_port}",
                endpoint_config(),
                max_in_flight=1,
                device="cpu",
                runtime=runtime,
                expected_worker_start_id="worker-start-a",
            )

        original = transport(original_runtime)
        same_generation = transport(same_generation_runtime)
        restarted = transport(restarted_runtime)
        try:
            await original.start()
            await original.close()
            assert receiver._target_generations == {  # type: ignore[attr-defined]
                "client:19001": "frontend-generation-a"
            }

            with pytest.raises(TransportError, match="INVALID_ARGUMENT"):
                await restarted.start()

            # Route removal/re-add within the same live Frontend process is
            # safe: its endpoint and runtime generation still identify the
            # descriptor owner retained by the Worker tombstone.
            await same_generation.start()
            await same_generation.close()
            assert receiver._target_generations == {  # type: ignore[attr-defined]
                "client:19001": "frontend-generation-a"
            }
        finally:
            await restarted.close()
            await same_generation.close()
            await original.close()
            await receiver.close()
            await original_runtime.close()
            await same_generation_runtime.close()
            await restarted_runtime.close()
            await worker_runtime.close()

    asyncio.run(scenario())


def test_commit_failure_retains_frontend_ownership_after_deregister(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        pair.receiver._session_close_grace_secs = 0.01  # type: ignore[attr-defined]

        async def fail_invalidation(
            target_session: str,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            del target_session, monotonic_deadline
            raise RuntimeError("native cache invalidation failed")

        monkeypatch.setattr(
            pair.worker_runtime,
            "invalidate_remote_session",
            fail_invalidation,
        )
        transport = pair.transport
        transport_ref = weakref.ref(transport)
        slab_ref = weakref.ref(transport._arena.slab)  # type: ignore[attr-defined]
        try:
            with pytest.raises(TransportError, match="arena is retained") as captured:
                await transport.close()
            assert captured.value.retryable is False
            assert pair.client_runtime.quarantine_reason is not None
            assert not transport._registered  # type: ignore[attr-defined]
            assert not pair.link.regions["client:19001"]
            assert transport._arena.slab.numel() > 0  # type: ignore[attr-defined]
            pair.transport = None  # type: ignore[assignment]
            del transport
            gc.collect()
            assert transport_ref() is not None
            assert slab_ref() is not None
        finally:
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_deregister_failure_keeps_target_gated_and_registered_arena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        pair.receiver._session_close_grace_secs = 0.01  # type: ignore[attr-defined]

        async def fail_deregister(
            tensor: torch.Tensor,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            del tensor, monotonic_deadline
            raise RuntimeError("native deregistration failed")

        monkeypatch.setattr(pair.client_runtime, "unregister_tensor", fail_deregister)
        try:
            with pytest.raises(TransportError, match="arena is retained") as captured:
                await pair.transport.close()
            assert captured.value.retryable is False
            assert pair.client_runtime.quarantine_reason is not None
            assert pair.transport._registered  # type: ignore[attr-defined]
            assert pair.link.regions["client:19001"]
            session = next(iter(pair.receiver._sessions.values()))  # type: ignore[attr-defined]
            assert session.close_prepared
            assert pair.receiver._target_close_gates == {  # type: ignore[attr-defined]
                "client:19001": {session.client_epoch}
            }
        finally:
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_lost_prepare_ack_keeps_target_gated_and_arena_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        pair.receiver._session_close_grace_secs = 0.01  # type: ignore[attr-defined]
        original_close = pair.transport._close_session  # type: ignore[attr-defined]

        async def lose_prepare_ack(payload: bytes, **kwargs: object) -> bytes:
            response = await original_close(payload, **kwargs)
            request = decode_close_session(payload, "worker-start-a")
            if request.phase == "prepare":
                raise RuntimeError("lost prepare acknowledgement")
            return response

        monkeypatch.setattr(pair.transport, "_close_session", lose_prepare_ack)
        try:
            with pytest.raises(TransportError, match="arena is retained"):
                await pair.transport.close()
            session = next(iter(pair.receiver._sessions.values()))  # type: ignore[attr-defined]
            assert session.close_prepared
            assert pair.transport._registered  # type: ignore[attr-defined]
            assert pair.link.regions["client:19001"]
            assert pair.worker_runtime.invalidated_sessions == []
            assert pair.receiver._target_close_gates == {  # type: ignore[attr-defined]
                "client:19001": {session.client_epoch}
            }
        finally:
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_lost_commit_ack_retains_deregistered_arena_after_worker_retire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        original_close = pair.transport._close_session  # type: ignore[attr-defined]

        async def lose_commit_ack(payload: bytes, **kwargs: object) -> bytes:
            response = await original_close(payload, **kwargs)
            request = decode_close_session(payload, "worker-start-a")
            if request.phase == "commit":
                raise RuntimeError("lost commit acknowledgement")
            return response

        monkeypatch.setattr(pair.transport, "_close_session", lose_commit_ack)
        try:
            with pytest.raises(TransportError, match="arena is retained"):
                await pair.transport.close()
            assert not pair.transport._registered  # type: ignore[attr-defined]
            assert not pair.link.regions["client:19001"]
            assert pair.transport._arena.slab.numel() > 0  # type: ignore[attr-defined]
            assert pair.worker_runtime.invalidated_sessions == ["client:19001"]
            assert not pair.receiver._sessions  # type: ignore[attr-defined]
            assert not pair.receiver._target_close_gates  # type: ignore[attr-defined]
            assert pair.client_runtime.quarantine_reason is not None
        finally:
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_worker_shutdown_keeps_close_session_listener_until_frontend_ack() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        shutdown = asyncio.create_task(pair.receiver.close())
        try:
            await asyncio.sleep(0.05)
            assert not shutdown.done()
            assert pair.receiver._server is not None  # type: ignore[attr-defined]

            await pair.transport.close()
            await shutdown

            assert pair.worker_runtime.invalidated_sessions == ["client:19001"]
            assert pair.worker_runtime.quarantine_reason is None
            assert not pair.link.regions["worker:19002"]
        finally:
            if not shutdown.done():
                shutdown.cancel()
            await asyncio.gather(shutdown, return_exceptions=True)
            await pair.transport.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_worker_shutdown_timeout_quarantines_and_retains_registered_arena() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        pair.receiver._session_close_grace_secs = 0.01  # type: ignore[attr-defined]
        await pair.receiver.close()

        assert pair.worker_runtime.quarantine_reason is not None
        assert pair.receiver._sessions  # type: ignore[attr-defined]
        assert pair.receiver._registered  # type: ignore[attr-defined]
        assert pair.link.regions["worker:19002"]

        with pytest.raises(TransportError, match="arena is retained"):
            await pair.transport.close()
        pair.buffers.close()
        await pair.client_runtime.close()
        await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_frontend_close_barrier_survives_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        invalidation_started = asyncio.Event()
        allow_invalidation = asyncio.Event()
        original_invalidate = pair.worker_runtime.invalidate_remote_session

        async def delayed_invalidation(
            target_session: str,
            *,
            monotonic_deadline: float = math.inf,
        ) -> None:
            invalidation_started.set()
            await allow_invalidation.wait()
            await original_invalidate(
                target_session,
                monotonic_deadline=monotonic_deadline,
            )

        monkeypatch.setattr(
            pair.worker_runtime,
            "invalidate_remote_session",
            delayed_invalidation,
        )
        caller = asyncio.create_task(pair.transport.close())
        try:
            await invalidation_started.wait()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert pair.transport._close_task is not None  # type: ignore[attr-defined]
            assert not pair.transport._close_task.done()  # type: ignore[attr-defined]

            allow_invalidation.set()
            await pair.transport.close()
            assert pair.client_runtime.quarantine_reason is None
            assert not pair.link.regions["client:19001"]
        finally:
            allow_invalidation.set()
            await pair.receiver.close()
            pair.buffers.close()
            await pair.client_runtime.close()
            await pair.worker_runtime.close()

    asyncio.run(scenario())


def test_cancelling_at_the_admission_frame_releases_the_worker_slot() -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        slot = pair.transport._arena.take()  # type: ignore[attr-defined]
        assert slot is not None
        slot.hidden_states[:2].zero_()
        slot.expert_ids[:2].zero_()
        slot.routing_weights[:2].zero_()
        request = TransferExecutePull(
            sequence=77,
            client_epoch=pair.transport._client_epoch,  # type: ignore[attr-defined]
            session_nonce=pair.transport._require_worker_session_nonce(),  # type: ignore[attr-defined]
            expected_worker_start_id="worker-start-a",
            client_slot_index=slot.index,
            client_slot_generation=slot.generation,
            layer_id=2,
            topology_version=11,
            token_count=2,
            timeout_micros=2_000_000,
            distinct_expert_ids=(0,),
        )
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{pair.receiver.bound_port}")
        execute = channel.unary_stream(
            "/ek.worker.v2.TransferEngineComputationService/ExecutePull",
            request_serializer=lambda payload: payload,
            response_deserializer=lambda payload: payload,
        )
        call = execute(
            encode_execute_pull_request(request, endpoint_config()),
            timeout=3,
            wait_for_ready=False,
        )
        try:
            admitted = await call.read()
            decode_admitted(admitted, request.sequence)
            call.cancel()
            for _ in range(200):
                session = pair.receiver._sessions[request.client_epoch]  # type: ignore[attr-defined]
                if (
                    pair.receiver.pending_count == 0
                    and pair.receiver.active_count == 0
                    and session.idle
                ):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("cancelled admission did not release its Worker slot")
            assert all(
                not worker_slot.in_use
                for worker_slot in pair.receiver._arena.slots  # type: ignore[attr-defined]
            )
        finally:
            pair.transport._arena.put(slot)  # type: ignore[attr-defined]
            await channel.close()
            await pair.close()

    asyncio.run(scenario())


def test_success_terminal_is_acquired_before_waiting_for_stream_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BlockedTrailingEofCall:
        def __init__(self) -> None:
            self.read_count = 0
            self.trailing_started = asyncio.Event()

        async def read(self) -> object:
            self.read_count += 1
            if self.read_count == 1:
                return encode_admitted(1)
            if self.read_count == 2:
                return encode_success(1)
            self.trailing_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def cancel(self) -> bool:
            return True

    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        call = _BlockedTrailingEofCall()
        monkeypatch.setattr(
            pair.transport,
            "_execute",
            lambda *args, **kwargs: call,
        )
        submission = asyncio.create_task(
            pair.transport.execute(
                worker_batch(),
                torch.empty((2, 3), dtype=torch.float32),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        try:
            await asyncio.wait_for(call.trailing_started.wait(), 2)
            assert pair.client_runtime.acquire_count == 1
            submission.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submission
            assert pair.client_runtime.quarantine_reason is None
            assert not pair.transport._arena.slots[0].in_use  # type: ignore[attr-defined]
        finally:
            if not submission.done():
                submission.cancel()
                await asyncio.gather(submission, return_exceptions=True)
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["copy", "record", "synchronize"])
def test_unprovable_input_staging_retains_the_caller_batch_and_poisoned_graph(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        batch = worker_batch()
        event = _InjectedCudaEvent(fail_record=1 if failure == "record" else None)
        pair.transport._arena.slots[0].copy_event = event  # type: ignore[attr-defined]
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())

        original_copy = client_module._copy_tensor  # type: ignore[attr-defined]
        copy_count = 0

        def injected_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
            nonlocal copy_count
            copy_count += 1
            if failure == "copy" and copy_count == 1:
                raise RuntimeError("injected CUDA input copy failure")
            original_copy(destination, source)

        monkeypatch.setattr(client_module, "_copy_tensor", injected_copy)
        if failure == "synchronize":

            async def fail_wait_event(
                cuda_event: torch.cuda.Event,
                *,
                monotonic_deadline: float,
            ) -> None:
                del cuda_event, monotonic_deadline
                pair.client_runtime.quarantine("injected CUDA event synchronize failure")
                raise TransportError(
                    TransportErrorCode.UNAVAILABLE,
                    retryable=False,
                    diagnostic="injected CUDA event synchronize failure",
                )

            monkeypatch.setattr(pair.client_runtime, "wait_event", fail_wait_event)
        try:
            with pytest.raises(TransportError, match="process must restart") as caught:
                await pair.transport.execute(
                    batch,
                    torch.empty((2, 3), dtype=torch.float32),
                    monotonic_deadline=time.monotonic() + 5,
                )
            assert not caught.value.retryable
            assert not caught.value.unsafe_output
            assert pair.client_runtime.quarantine_reason is not None
            retained = client_module._UNSAFE_STAGING_GRAPHS[-1]  # type: ignore[attr-defined]
            assert batch in retained
            assert pair.transport in retained
            assert pair.client_runtime in retained
        finally:
            await pair.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["copy", "record", "synchronize"])
def test_unprovable_output_staging_quarantines_the_output_pool_lease(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        pair = await start_pair(max_in_flight=1)
        output_pool = OutputPool(
            max_batch_tokens=4,
            hidden_dim=3,
            dtype=torch.float32,
            device="cpu",
            capacity=1,
        )
        event = _InjectedCudaEvent(fail_record=2 if failure == "record" else None)
        pair.transport._arena.slots[0].copy_event = event  # type: ignore[attr-defined]
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())

        original_copy = client_module._copy_tensor  # type: ignore[attr-defined]
        copy_count = 0

        def injected_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
            nonlocal copy_count
            copy_count += 1
            if failure == "copy" and copy_count == 3:
                raise RuntimeError("injected CUDA output copy failure")
            original_copy(destination, source)

        monkeypatch.setattr(client_module, "_copy_tensor", injected_copy)
        if failure == "synchronize":
            wait_count = 0
            original_wait = pair.client_runtime.wait_event

            async def fail_second_wait(
                cuda_event: torch.cuda.Event,
                *,
                monotonic_deadline: float,
            ) -> None:
                nonlocal wait_count
                wait_count += 1
                if wait_count == 2:
                    pair.client_runtime.quarantine("injected CUDA event synchronize failure")
                    raise TransportError(
                        TransportErrorCode.UNAVAILABLE,
                        retryable=False,
                        diagnostic="injected CUDA event synchronize failure",
                    )
                await original_wait(
                    cuda_event,
                    monotonic_deadline=monotonic_deadline,
                )

            monkeypatch.setattr(pair.client_runtime, "wait_event", fail_second_wait)

        leased_tensor: torch.Tensor | None = None
        try:
            with pytest.raises(TransportError, match="process must restart") as caught:
                async with output_pool.lease(monotonic_deadline=float("inf")) as lease:
                    leased_tensor = lease.tensor
                    submission = asyncio.create_task(
                        pair.transport.execute(
                            worker_batch(),
                            lease.tensor[:2],
                            monotonic_deadline=time.monotonic() + 5,
                        )
                    )
                    received = await pair.receiver.receive()
                    await complete(received)
                    await submission
            assert not caught.value.retryable
            assert caught.value.unsafe_output
            assert pair.client_runtime.quarantine_reason is not None
            assert not output_pool._available  # type: ignore[attr-defined]
            assert len(output_pool._quarantined) == 1  # type: ignore[attr-defined]
            with pytest.raises(TransportError, match="pool is unavailable"):
                async with output_pool.lease(monotonic_deadline=float("inf")):
                    pass
            retained = client_module._UNSAFE_STAGING_GRAPHS[-1]  # type: ignore[attr-defined]
            assert pair.transport in retained
            assert leased_tensor is not None
            assert any(
                isinstance(item, torch.Tensor) and item.data_ptr() == leased_tensor.data_ptr()
                for item in retained
            )
        finally:
            await output_pool.close()
            await pair.close()

    asyncio.run(scenario())
