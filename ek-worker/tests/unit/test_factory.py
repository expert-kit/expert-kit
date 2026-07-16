"""Tests for concrete Worker component construction."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from expertkit_worker.config import WorkerConfig
from expertkit_worker.factory import _split_address, build_worker_application


def _config(cache_path: Path, *, backend: str = "torch") -> WorkerConfig:
    worker_device = "cpu" if backend == "ggml" else "cuda:0"
    document: dict[str, object] = {
        "model": {
            "instance_id": 7,
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
            "device_memory_limit": "1GiB",
        },
        "transport": {
            "max_pending_batches_per_device": 1,
            "grpc": {
                "listen": "127.0.0.1:50051",
                "advertise": "worker-0:50051",
            },
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
    if backend == "ggml":
        document["ggml"] = {"cpu_threads": 2}
    return WorkerConfig.model_validate(document)


def test_split_address_handles_ipv4_hostnames_and_ipv6() -> None:
    assert _split_address("127.0.0.1:5000") == ("127.0.0.1", 5000)
    assert _split_address("worker.local:5001") == ("worker.local", 5001)
    assert _split_address("[2001:db8::1]:5002") == ("2001:db8::1", 5002)


def test_factory_rejects_unimplemented_optional_backend(tmp_path: Path) -> None:
    async def scenario() -> None:
        with pytest.raises(NotImplementedError, match="ggml Backend"):
            await build_worker_application(_config(tmp_path, backend="ggml"))

    asyncio.run(scenario())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_factory_builds_and_closes_torch_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        application = await build_worker_application(_config(tmp_path))
        await application.close()

    asyncio.run(scenario())
