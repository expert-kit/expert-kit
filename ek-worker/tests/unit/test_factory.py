"""Tests for concrete Worker component construction."""

from __future__ import annotations

import asyncio
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch
from expertkit_transport.controller import ResolvedDefaultInstance

from expertkit_worker.config import WorkerConfig
from expertkit_worker.factory import build_worker_application
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
    monkeypatch.setattr(
        "expertkit_worker.factory._memory_info",
        lambda _device: (2**40, 2**40),
    )

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
