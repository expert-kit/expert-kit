"""Tests for concrete Worker component construction."""

from __future__ import annotations

import asyncio
import builtins
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from expertkit_transport.controller import ResolvedDefaultInstance
from expertkit_transport.transports.base import BatchBufferConfig
from expertkit_transport.transports.grpc.worker_buffers import GrpcWorkerBatchBuffers

from expertkit_worker.config import WorkerConfig
from expertkit_worker.execution import AsyncExecutionSlot, CpuExecutionSlot
from expertkit_worker.factory import (
    _async_wiring,
    _create_device_wiring,
    build_worker_application,
)
from expertkit_worker.weights import DirectIOWeightDiskCache


def _config(
    cache_path: Path,
    *,
    backend: str = "torch",
    instance_id: int | None = 7,
) -> WorkerConfig:
    worker_device = "cpu" if backend == "ggml" else "cuda:0"
    document: dict[str, object] = {
        "model": {
            "name": "fixture/model",
            "weight_version": "test",
            "num_layers": 2,
            "experts_per_layer": 4,
            "hidden_dim": 4,
            "expert_intermediate_dim": 8,
            "top_k": 2,
            "activation_dtype": "fp16",
            "weight_dtype": "fp16",
        },
        "worker": {
            "id": "worker-0",
            "backend": backend,
            "device": worker_device,
            "max_batch_tokens": 4,
            "max_active_batches_per_device": 1,
            "device_memory_limit": "513MiB" if backend == "fused" else "1GiB",
        },
        "transport": {
            "type": "grpc",
            "max_pending_batches_per_device": 1,
            "listen": "127.0.0.1:50051",
            "advertise": "worker-0:50051",
        },
        "controller": {"endpoint": "127.0.0.1:50050"},
        "weight_manager": {
            "max_concurrent_loads": 2,
            "disk_cache": {"path": str(cache_path)},
            "peer": {
                "listen": "[::1]:50052",
                "advertise": "http://worker-0:50052",
            },
            "weight_server_endpoint": "http://127.0.0.1:50053",
        },
    }
    if instance_id is not None:
        document["model"]["instance_id"] = instance_id
    if backend == "ggml":
        document["worker"]["ggml"] = {"cpu_threads": 2}
    return WorkerConfig.model_validate(document)


async def _resolve_instance(
    endpoint: str,
    *,
    requested_instance_id: int | None,
    timeout_seconds: float,
) -> ResolvedDefaultInstance:
    assert endpoint == "127.0.0.1:50050"
    assert requested_instance_id in (None, 7)
    assert timeout_seconds == 10
    return ResolvedDefaultInstance(7, "fixture/model", "default")


def test_factory_builds_and_closes_ggml_worker(tmp_path: Path) -> None:
    try:
        version("ggml-python")
    except PackageNotFoundError:
        pytest.skip("the GGML extra is not installed")

    async def scenario() -> None:
        application = await build_worker_application(
            _config(tmp_path, backend="ggml"),
            instance_resolver=_resolve_instance,
        )
        await application.close()

    asyncio.run(scenario())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_factory_builds_and_closes_fused_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        application = await build_worker_application(
            _config(tmp_path, backend="fused"),
            instance_resolver=_resolve_instance,
        )
        await application.close()

    asyncio.run(scenario())


def test_factory_probes_direct_io_before_returning_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = False

    async def fail_initialize(_cache: DirectIOWeightDiskCache) -> None:
        nonlocal initialized
        initialized = True
        raise OSError("direct I/O probe failed")

    monkeypatch.setattr(DirectIOWeightDiskCache, "initialize", fail_initialize)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)

    async def scenario() -> None:
        with pytest.raises(OSError, match="direct I/O probe failed"):
            await build_worker_application(
                _config(tmp_path),
                instance_resolver=_resolve_instance,
            )

    asyncio.run(scenario())
    assert initialized is True


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_factory_builds_and_closes_torch_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        application = await build_worker_application(
            _config(tmp_path),
            instance_resolver=_resolve_instance,
        )
        await application.close()

    asyncio.run(scenario())


def test_factory_resolves_an_omitted_instance_before_device_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []

    async def fail_resolution(
        _endpoint: str,
        *,
        requested_instance_id: int | None,
        timeout_seconds: float,
    ) -> ResolvedDefaultInstance:
        calls.append(requested_instance_id)
        assert timeout_seconds == 10
        raise RuntimeError("Controller resolution failed")

    monkeypatch.setattr(
        "expertkit_worker.factory.torch_dtype",
        lambda _dtype: pytest.fail("device setup started before instance resolution"),
    )

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="Controller resolution failed"):
            await build_worker_application(
                _config(tmp_path, instance_id=None),
                instance_resolver=fail_resolution,
            )

    asyncio.run(scenario())
    assert calls == [None]


def _buffer_config() -> BatchBufferConfig:
    return BatchBufferConfig(
        max_batch_tokens=2,
        hidden_dim=4,
        top_k=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_cpu_wiring_builds_cpu_slot() -> None:
    wiring = _create_device_wiring("cpu")
    spec = _buffer_config()

    slot = wiring.create_slot(spec, GrpcWorkerBatchBuffers(spec))

    assert isinstance(slot, CpuExecutionSlot)
    assert wiring.runtime.device == torch.device("cpu")
    slot.close()


class _FakeAsyncRuntime:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.current_device_set = False

    def device_context(self):
        from contextlib import nullcontext

        return nullcontext()

    def create_stream(self, *, priority: int = 0) -> object:
        assert priority == 0
        return object()

    def create_event(self, *, enable_timing: bool = False) -> object:
        raise AssertionError(f"timing event unexpectedly requested: {enable_timing}")

    def set_current_device(self) -> None:
        self.current_device_set = True


def test_async_wiring_captures_runtime_in_slot_factory() -> None:
    runtime = _FakeAsyncRuntime()
    wiring = _async_wiring(runtime)  # type: ignore[arg-type]
    spec = _buffer_config()

    slot = wiring.create_slot(
        spec,
        GrpcWorkerBatchBuffers(spec),
        enable_device_timing=False,
    )

    assert runtime.current_device_set is True
    assert isinstance(slot, AsyncExecutionSlot)
    slot.close()


def test_cpu_and_cuda_selection_do_not_import_ascend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import expertkit_worker.factory as factory

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        if name == "expertkit_worker.device.ascend":
            raise AssertionError("Ascend runtime imported for a non-NPU device")
        return real_import(name, *args, **kwargs)

    class FakeCudaRuntime:
        def __init__(self, device: torch.device) -> None:
            self.device = device

    cuda_wiring = object()
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(factory, "CudaWorkerRuntime", FakeCudaRuntime)
    monkeypatch.setattr(factory, "_async_wiring", lambda _runtime: cuda_wiring)

    cpu_wiring = _create_device_wiring("cpu")
    assert isinstance(cpu_wiring.runtime, factory.CpuWorkerRuntime)
    assert _create_device_wiring("cuda:0") is cuda_wiring


def test_npu_selection_lazily_loads_ascend_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import expertkit_worker.factory as factory

    selected: list[object] = []

    class FakeAscendRuntime:
        def __init__(self, device: object) -> None:
            selected.append(device)

    module = ModuleType("expertkit_worker.device.ascend")
    module.AscendWorkerRuntime = FakeAscendRuntime  # type: ignore[attr-defined]
    wiring = object()
    monkeypatch.setitem(sys.modules, "expertkit_worker.device.ascend", module)
    monkeypatch.setattr(factory.torch, "device", lambda name: f"parsed:{name}")
    monkeypatch.setattr(factory, "_async_wiring", lambda _runtime: wiring)

    assert _create_device_wiring("npu:2") is wiring
    assert selected == ["parsed:npu:2"]
