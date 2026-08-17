"""Fixed arena and native runtime lifetime checks."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from expertkit_transport.errors import TransportError
from expertkit_transport.transports.base import WorkerEndpointConfig
from expertkit_transport.transports.transfer_engine import runtime as runtime_module
from expertkit_transport.transports.transfer_engine.arena import TransferArena
from expertkit_transport.transports.transfer_engine.runtime import (
    TransferEngineRuntime,
    TransferEngineRuntimeConfig,
)
from expertkit_transport.transports.transfer_engine.session import TransferEngineSession


def endpoint_config() -> WorkerEndpointConfig:
    return WorkerEndpointConfig(7, 4, 8, 4, 3, 2, torch.float32)


def test_runtime_rejects_an_unknown_native_protocol() -> None:
    with pytest.raises(ValueError, match="protocol must be"):
        TransferEngineRuntimeConfig(
            segment_name="127.0.0.1:19000",
            metadata_server="P2PHANDSHAKE",
            protocol="auto",
            device="cpu",
        )


def test_runtime_rejects_nvlink_on_a_cpu_device_and_nonempty_hint() -> None:
    with pytest.raises(ValueError, match="require a CUDA"):
        TransferEngineRuntimeConfig(
            segment_name="127.0.0.1:19000",
            metadata_server="P2PHANDSHAKE",
            protocol="nvlink_intra",
            device="cpu",
        )
    with pytest.raises(ValueError, match="transport_hint must be empty"):
        TransferEngineRuntimeConfig(
            segment_name="127.0.0.1:19000",
            metadata_server="P2PHANDSHAKE",
            protocol="tcp",
            device="cpu",
            transport_hint="tcp",
        )


def test_runtime_requires_explicit_cuda_rdma_configuration() -> None:
    common = {
        "segment_name": "192.0.2.11",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "rdma",
        "device_name": "mlx5_0",
    }
    with pytest.raises(ValueError, match="enable_experimental_rdma=True"):
        TransferEngineRuntimeConfig(device="cuda:0", **common)
    with pytest.raises(ValueError, match="indexed CUDA device"):
        TransferEngineRuntimeConfig(
            device="cpu",
            enable_experimental_rdma=True,
            **common,
        )
    with pytest.raises(ValueError, match="non-empty device_name"):
        TransferEngineRuntimeConfig(
            segment_name="192.0.2.11",
            metadata_server="P2PHANDSHAKE",
            protocol="rdma",
            device="cuda:0",
            enable_experimental_rdma=True,
        )
    with pytest.raises(ValueError, match="metadata_server=P2PHANDSHAKE"):
        TransferEngineRuntimeConfig(
            segment_name="192.0.2.11",
            metadata_server="etcd://192.0.2.12:2379",
            protocol="rdma",
            device="cuda:0",
            device_name="mlx5_0",
            enable_experimental_rdma=True,
        )
    config = TransferEngineRuntimeConfig(
        device="cuda:0",
        enable_experimental_rdma=True,
        **common,
    )
    assert config.protocol == "rdma"
    assert config.device == torch.device("cuda:0")

    with pytest.raises(ValueError, match="valid only when protocol is rdma"):
        TransferEngineRuntimeConfig(
            segment_name="127.0.0.1",
            metadata_server="P2PHANDSHAKE",
            protocol="tcp",
            device="cpu",
            enable_experimental_rdma=True,
        )


def test_rdma_native_capabilities_fail_closed_independently() -> None:
    config = TransferEngineRuntimeConfig(
        segment_name="192.0.2.11",
        metadata_server="P2PHANDSHAKE",
        protocol="rdma",
        device="cuda:0",
        device_name="mlx5_0",
        enable_experimental_rdma=True,
    )
    capabilities = {
        "EK_SAFE_TERMINAL_BATCH_SYNC": True,
        "EK_HAS_GPUDIRECT_ACQUIRE": True,
        "EK_FORCE_CONFIGURED_RDMA_TRANSPORT": True,
        "EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION": True,
    }
    runtime_module._validate_native_capabilities(  # type: ignore[attr-defined]
        SimpleNamespace(**capabilities),
        config,
    )
    for missing, diagnostic in (
        ("EK_FORCE_CONFIGURED_RDMA_TRANSPORT", "force the configured RDMA backend"),
        (
            "EK_DRAINED_RDMA_REMOTE_DESCRIPTOR_INVALIDATION",
            "drained RDMA remote-descriptor invalidation",
        ),
    ):
        incomplete = dict(capabilities)
        incomplete[missing] = False
        with pytest.raises(RuntimeError, match=diagnostic):
            runtime_module._validate_native_capabilities(  # type: ignore[attr-defined]
                SimpleNamespace(**incomplete),
                config,
            )


def test_configured_backend_query_requires_an_exact_native_result() -> None:
    engine = SimpleNamespace(get_configured_backend=lambda: "rdma")
    query = runtime_module._configured_backend_query(  # type: ignore[attr-defined]
        engine,
        "rdma",
    )
    assert query is not None
    assert (
        runtime_module._validate_configured_backend(  # type: ignore[attr-defined]
            query(),
            "rdma",
        )
        == "rdma"
    )
    with pytest.raises(RuntimeError, match="expected rdma, got 'tcp'"):
        runtime_module._validate_configured_backend(  # type: ignore[attr-defined]
            "tcp",
            "rdma",
        )
    with pytest.raises(RuntimeError, match="cannot report"):
        runtime_module._configured_backend_query(  # type: ignore[attr-defined]
            object(),
            "rdma",
        )
    assert (
        runtime_module._configured_backend_query(  # type: ignore[attr-defined]
            object(),
            "tcp",
        )
        is None
    )


def test_remote_invalidation_api_is_backend_specific() -> None:
    engine = SimpleNamespace(
        invalidate_drained_nvlink_intra_segment=lambda target: target,
        invalidate_drained_rdma_segment=lambda target: target,
    )
    _, nvlink_subject = runtime_module._remote_invalidation_api(  # type: ignore[attr-defined]
        engine,
        "nvlink_intra",
    )
    _, rdma_subject = runtime_module._remote_invalidation_api(  # type: ignore[attr-defined]
        engine,
        "rdma",
    )
    assert "intra-NVLink" in nvlink_subject
    assert "RDMA" in rdma_subject
    with pytest.raises(RuntimeError, match="requires a forced"):
        runtime_module._remote_invalidation_api(  # type: ignore[attr-defined]
            engine,
            "tcp",
        )
    with pytest.raises(RuntimeError, match="lacks mooncake drained rdma"):
        runtime_module._remote_invalidation_api(  # type: ignore[attr-defined]
            object(),
            "rdma",
        )


@pytest.mark.parametrize("name", ["MC_USE_TENT", "MC_USE_TEV1"])
def test_runtime_rejects_any_tent_environment_value(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setenv(name, "0")
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
            ),
            engine=_BlockingEngine(),
        )
        with pytest.raises(RuntimeError, match=r"does not support TENT.*unset"):
            await runtime.start()
        assert not runtime._registrations  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["MC_USE_TENT", "MC_USE_TEV1"])
def test_experimental_rdma_does_not_bypass_tent_rejection(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setenv(name, "0")
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="192.0.2.11",
                metadata_server="P2PHANDSHAKE",
                protocol="rdma",
                device="cuda:0",
                device_name="mlx5_0",
                enable_experimental_rdma=True,
            ),
            engine=_BlockingEngine(),
        )
        with pytest.raises(RuntimeError, match=r"does not support TENT.*unset"):
            await runtime.start()
        assert not runtime._registrations  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


def test_arena_uses_one_contiguous_allocation_with_typed_slot_views() -> None:
    arena = TransferArena(endpoint_config(), device="cpu", capacity=2)
    descriptor = arena.descriptor

    descriptor.validate(endpoint_config())
    assert arena.slab.data_ptr() == descriptor.base_address
    assert descriptor.addresses(0) == tuple(
        tensor.data_ptr()
        for tensor in (
            arena.slots[0].hidden_states,
            arena.slots[0].expert_ids,
            arena.slots[0].routing_weights,
            arena.slots[0].partial_output,
        )
    )
    assert descriptor.addresses(1) == tuple(
        tensor.data_ptr()
        for tensor in (
            arena.slots[1].hidden_states,
            arena.slots[1].expert_ids,
            arena.slots[1].routing_weights,
            arena.slots[1].partial_output,
        )
    )

    first = arena.take()
    assert first is arena.slots[0]
    assert first.generation == 1
    arena.put(first)
    again = arena.take()
    assert again is first
    assert again.generation == 2
    arena.put(again)
    arena.close()


def test_arena_rejects_capacity_above_the_wire_protocol_limit() -> None:
    with pytest.raises(ValueError, match="capacity cannot exceed 4096"):
        TransferArena(endpoint_config(), device="cpu", capacity=4097)


def test_session_rejects_two_active_sequences_for_the_same_client_slot() -> None:
    arena = TransferArena(endpoint_config(), device="cpu", capacity=2)
    session = TransferEngineSession(
        client_epoch="frontend-epoch",
        session_nonce="worker-nonce",
        target_session_id="frontend:19001",
        target_runtime_generation="frontend-generation",
        backend="tcp",
        arena=arena.descriptor,
    )
    session.claim(sequence=1, slot_index=0, generation=1)
    with pytest.raises(Exception, match="slot is already active"):
        session.claim(sequence=2, slot_index=0, generation=2)
    session.release(1)
    session.claim(sequence=2, slot_index=0, generation=2)
    session.release(2)
    arena.close()


class _BlockingEngine:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.thread_names: list[str] = []

    def initialize(self, *args: object) -> int:
        del args
        self.thread_names.append(threading.current_thread().name)
        return 0

    def get_rpc_port(self) -> int:
        return 19001

    def register_memory(self, address: int, length: int) -> int:
        assert address > 0 and length > 0
        return 0

    def unregister_memory(self, address: int) -> int:
        assert address > 0
        return 0

    def batch_transfer_sync_read(
        self,
        target: str,
        local: list[int],
        remote: list[int],
        lengths: list[int],
        hint: str,
    ) -> int:
        del target, local, remote, lengths, hint
        self.thread_names.append(threading.current_thread().name)
        self.started.set()
        assert self.release.wait(5)
        return 0


class _BackendReportingEngine(_BlockingEngine):
    def __init__(self, backend: str) -> None:
        super().__init__()
        self.backend = backend
        self.invalidated_sessions: list[str] = []

    def get_configured_backend(self) -> str:
        return self.backend

    def invalidate_drained_rdma_segment(self, target_session: str) -> int:
        assert target_session
        self.invalidated_sessions.append(target_session)
        return 0


def test_runtime_rejects_native_backend_mismatch_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="192.0.2.11",
                metadata_server="P2PHANDSHAKE",
                protocol="rdma",
                device="cuda:0",
                device_name="mlx5_0",
                enable_experimental_rdma=True,
            ),
            engine=_BackendReportingEngine("tcp"),
        )
        with pytest.raises(RuntimeError, match="expected rdma, got 'tcp'"):
            await runtime.start()
        assert not runtime._registrations  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_uses_exact_rdma_backend_and_native_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
        engine = _BackendReportingEngine("rdma")
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="192.0.2.11",
                metadata_server="P2PHANDSHAKE",
                protocol="rdma",
                device="cuda:0",
                device_name="mlx5_0",
                enable_experimental_rdma=True,
            ),
            engine=engine,
        )
        await runtime.start()
        assert runtime.backend == "rdma"
        await runtime.invalidate_remote_session("192.0.2.10:19001")
        assert engine.invalidated_sessions == ["192.0.2.10:19001"]
        await runtime.close()

    asyncio.run(scenario())


class _BlockingRegistrationEngine(_BlockingEngine):
    def __init__(self) -> None:
        super().__init__()
        self.registration_started = threading.Event()
        self.allow_registration = threading.Event()

    def register_memory(self, address: int, length: int) -> int:
        assert address > 0 and length > 0
        self.registration_started.set()
        assert self.allow_registration.wait(5)
        return 0


class _BlockingUnregistrationEngine(_BlockingEngine):
    def __init__(self) -> None:
        super().__init__()
        self.unregistration_started = threading.Event()
        self.allow_unregistration = threading.Event()

    def unregister_memory(self, address: int) -> int:
        assert address > 0
        self.unregistration_started.set()
        assert self.allow_unregistration.wait(5)
        return 0


class _FailingRegistrationEngine(_BlockingEngine):
    def register_memory(self, address: int, length: int) -> int:
        assert address > 0 and length > 0
        return -1


class _FailingUnregistrationEngine(_BlockingEngine):
    def unregister_memory(self, address: int) -> int:
        assert address > 0
        return -1


class _ExplodingTransferEngine(_BlockingEngine):
    def batch_transfer_sync_read(
        self,
        target: str,
        local: list[int],
        remote: list[int],
        lengths: list[int],
        hint: str,
    ) -> int:
        del target, local, remote, lengths, hint
        raise RuntimeError("injected native read exception")

    def batch_transfer_sync_write(
        self,
        target: str,
        local: list[int],
        remote: list[int],
        lengths: list[int],
        hint: str,
    ) -> int:
        del target, local, remote, lengths, hint
        raise RuntimeError("injected native write exception")


def test_failed_registration_quarantines_and_retains_the_tensor() -> None:
    async def scenario() -> None:
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
            ),
            engine=_FailingRegistrationEngine(),
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        with pytest.raises(Exception, match="native status -1"):
            await runtime.register_tensor(slab)
        with pytest.raises(Exception, match="quarantined"):
            runtime.ensure_healthy()
        assert runtime._pending_registrations[slab.data_ptr()][0] is slab  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


def test_failed_unregistration_quarantines_and_retains_the_tensor() -> None:
    async def scenario() -> None:
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
            ),
            engine=_FailingUnregistrationEngine(),
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        await runtime.register_tensor(slab)
        with pytest.raises(Exception, match="native status -1"):
            await runtime.unregister_tensor(slab)
        with pytest.raises(Exception, match="quarantined"):
            runtime.ensure_healthy()
        assert runtime._registrations[slab.data_ptr()][0] is slab  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["read", "write"])
def test_unexpected_native_batch_exception_quarantines_registered_storage(
    operation: str,
) -> None:
    async def scenario() -> None:
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
            ),
            engine=_ExplodingTransferEngine(),
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        await runtime.register_tensor(slab)

        transfer = runtime.batch_read if operation == "read" else runtime.batch_write
        with pytest.raises(RuntimeError, match=f"native {operation} exception"):
            await transfer(
                "peer:19002",
                (slab[:16],),
                (12345,),
                (16,),
                monotonic_deadline=time.monotonic() + 5,
            )

        with pytest.raises(TransportError, match="quarantined"):
            runtime.ensure_healthy()
        assert runtime._registrations[slab.data_ptr()][0] is slab  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


def test_cancelled_registration_commits_python_ownership_before_raising() -> None:
    async def scenario() -> None:
        engine = _BlockingRegistrationEngine()
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
                max_workers=1,
            ),
            engine=engine,
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        registration = asyncio.create_task(runtime.register_tensor(slab))
        assert await asyncio.to_thread(engine.registration_started.wait, 2)
        registration.cancel()
        registration.cancel()
        await asyncio.sleep(0)
        assert not registration.done()

        engine.allow_registration.set()
        with pytest.raises(asyncio.CancelledError):
            await registration
        assert runtime._registrations[slab.data_ptr()][0] is slab  # type: ignore[attr-defined]
        await runtime.unregister_tensor(slab)
        await runtime.close()

    asyncio.run(scenario())


def test_cancelled_unregistration_commits_python_release_before_raising() -> None:
    async def scenario() -> None:
        engine = _BlockingUnregistrationEngine()
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
                max_workers=1,
            ),
            engine=engine,
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        await runtime.register_tensor(slab)
        unregistration = asyncio.create_task(runtime.unregister_tensor(slab))
        assert await asyncio.to_thread(engine.unregistration_started.wait, 2)
        unregistration.cancel()
        unregistration.cancel()
        await asyncio.sleep(0)
        assert not unregistration.done()

        engine.allow_unregistration.set()
        with pytest.raises(asyncio.CancelledError):
            await unregistration
        assert slab.data_ptr() not in runtime._registrations  # type: ignore[attr-defined]
        await runtime.close()

    asyncio.run(scenario())


def test_cancel_waits_for_native_dma_before_registered_slab_can_be_reused() -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
                max_workers=1,
            ),
            engine=engine,
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        assert runtime.session_id == "127.0.0.1:19001"
        await runtime.register_tensor(slab)
        transfer = asyncio.create_task(
            runtime.batch_read(
                "peer:19002",
                (slab[:16],),
                (12345,),
                (16,),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        assert await asyncio.to_thread(engine.started.wait, 2)
        transfer.cancel()
        await asyncio.sleep(0)
        assert not transfer.done()

        engine.release.set()
        with pytest.raises(asyncio.CancelledError):
            await transfer
        await runtime.unregister_tensor(slab)
        await runtime.close()
        assert engine.thread_names
        assert all(name.startswith("expertkit-transfer-engine") for name in engine.thread_names)

    asyncio.run(scenario())


def test_runtime_close_waits_for_native_dma_and_rejects_new_operations() -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
                max_workers=1,
            ),
            engine=engine,
        )
        slab = torch.empty(1024, dtype=torch.uint8)
        await runtime.start()
        await runtime.register_tensor(slab)
        transfer = asyncio.create_task(
            runtime.batch_read(
                "peer:19002",
                (slab[:16],),
                (12345,),
                (16,),
                monotonic_deadline=time.monotonic() + 5,
            )
        )
        assert await asyncio.to_thread(engine.started.wait, 2)

        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(Exception, match="runtime is closing"):
            await runtime.batch_read(
                "peer:19002",
                (slab[:16],),
                (12345,),
                (16,),
                monotonic_deadline=time.monotonic() + 5,
            )

        engine.release.set()
        await transfer
        await closing
        assert not runtime._registrations  # type: ignore[attr-defined]

    asyncio.run(scenario())


def test_p2p_segment_with_explicit_port_is_not_appended_twice() -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="127.0.0.1:19000",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
            ),
            engine=engine,
        )
        await runtime.start()
        assert runtime.session_id == "127.0.0.1:19001"
        await runtime.close()

    asyncio.run(scenario())


def test_p2p_segment_formats_a_raw_ipv6_address() -> None:
    async def scenario() -> None:
        engine = _BlockingEngine()
        runtime = TransferEngineRuntime(
            TransferEngineRuntimeConfig(
                segment_name="2001:db8::1",
                metadata_server="P2PHANDSHAKE",
                protocol="tcp",
                device="cpu",
            ),
            engine=engine,
        )
        await runtime.start()
        assert runtime.session_id == "[2001:db8::1]:19001"
        await runtime.close()

    asyncio.run(scenario())
