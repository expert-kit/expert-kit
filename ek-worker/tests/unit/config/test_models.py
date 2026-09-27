"""Tests for Backend-specific Worker configuration constraints."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from expertkit_worker.config import (
    NcclTransportConfig,
    ShmTransportConfig,
    TransferEngineTransportConfig,
    WorkerConfig,
)


def _config(cache_path: Path, *, backend: str, device: str) -> dict[str, object]:
    return {
        "model": {
            "instance_id": 1,
            "name": "test",
            "weight_version": "v1",
            "num_layers": 1,
            "experts_per_layer": 4,
            "hidden_dim": 16,
            "expert_intermediate_dim": 32,
            "top_k": 2,
            "activation_dtype": "fp16",
            "weight_dtype": "fp16",
        },
        "worker": {
            "id": "worker-0",
            "backend": backend,
            "device": device,
            "device_memory_limit": "2GiB",
        },
        "transport": {
            "type": "grpc",
            "listen": "127.0.0.1:50051",
            "advertise": "worker:50051",
        },
        "controller": {"endpoint": "controller:50050"},
        "weight_manager": {
            "disk_cache": {"path": cache_path},
            "peer": {
                "listen": "127.0.0.1:50052",
                "advertise": "http://worker:50052",
            },
            "weight_server_endpoint": "http://weights:8080",
        },
    }


def test_ggml_requires_cpu_threads(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ggml", device="cpu")

    with pytest.raises(ValidationError, match=r"worker\.ggml configuration is required"):
        WorkerConfig.model_validate(raw)

    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["ggml"] = {"cpu_threads": 8}
    config = WorkerConfig.model_validate(raw)
    assert config.worker.ggml is not None
    assert config.worker.ggml.cpu_threads == 8


def test_non_ggml_backend_rejects_ggml_settings(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["ggml"] = {"cpu_threads": 8}

    with pytest.raises(ValidationError, match=r"valid only when worker\.backend is ggml"):
        WorkerConfig.model_validate(raw)


def test_fused_rejects_fp32(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="fused", device="cuda:1")
    model = raw["model"]
    assert isinstance(model, dict)
    model["activation_dtype"] = "fp32"

    with pytest.raises(ValidationError, match="supports only FP16 or BF16"):
        WorkerConfig.model_validate(raw)


def test_paths_must_be_absolute(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    weight_manager = raw["weight_manager"]
    assert isinstance(weight_manager, dict)
    weight_manager["disk_cache"] = {"path": "relative/cache"}

    with pytest.raises(ValidationError, match="path must be absolute"):
        WorkerConfig.model_validate(raw)


def test_device_memory_limit_must_be_positive(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["device_memory_limit"] = 0

    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig.model_validate(raw)


def test_torch_backend_accepts_indexed_npu_device(tmp_path: Path) -> None:
    config = WorkerConfig.model_validate(_config(tmp_path, backend="torch", device="npu:0"))

    assert config.worker.device == "npu:0"


@pytest.mark.parametrize("device", ["npu", "npu:x", "cuda", "cuda:x"])
def test_torch_backend_rejects_unindexed_or_invalid_accelerator(
    tmp_path: Path,
    device: str,
) -> None:
    with pytest.raises(ValidationError, match=r"cpu, cuda:<id> or npu:<id>"):
        WorkerConfig.model_validate(_config(tmp_path, backend="torch", device=device))


def test_fused_backend_remains_cuda_only(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match=r"requires cuda:<id>"):
        WorkerConfig.model_validate(_config(tmp_path, backend="fused", device="npu:0"))


def test_npu_worker_rejects_shm_transport(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="npu:0")
    raw["transport"] = {
        "type": "shm",
        "rpc_listen": "127.0.0.1:50051",
        "rpc_advertise": "worker:50051",
        "shared_memory_dir": "/dev/shm",
    }

    with pytest.raises(ValidationError, match="NPU workers currently require the gRPC transport"):
        WorkerConfig.model_validate(raw)


def test_gptq_quantization_requires_4_bit_grouped_weights(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    model = raw["model"]
    assert isinstance(model, dict)
    model["quantization"] = {"type": "gptq", "bits": 4, "group_size": 128}

    config = WorkerConfig.model_validate(raw)

    assert config.model.quantization is not None
    assert config.model.quantization.group_size == 128


def test_shm_transport_uses_only_notification_rpc_fields(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = {
        "type": "shm",
        "rpc_listen": "127.0.0.1:50051",
        "rpc_advertise": "worker:50051",
        "shared_memory_dir": "/dev/shm",
    }

    config = WorkerConfig.model_validate(raw)

    assert isinstance(config.transport, ShmTransportConfig)
    assert config.transport.max_pending_batches_per_device == 1
    assert config.transport.shared_memory_dir == "/dev/shm"


def test_nccl_transport_validates_static_process_group(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = {
        "type": "nccl",
        "control_listen": "127.0.0.1:50051",
        "control_advertise": "worker:50051",
        "rank": 1,
        "world_size": 2,
        "rendezvous_endpoint": "controller:29500",
        "group_name": "expert-kit",
    }

    config = WorkerConfig.model_validate(raw)

    assert isinstance(config.transport, NcclTransportConfig)
    assert config.transport.max_pending_batches_per_device == 1
    assert config.transport.rank == 1
    assert config.transport.world_size == 2

    raw["transport"]["rank"] = 2  # type: ignore[index]
    with pytest.raises(ValidationError, match="rank must be less than"):
        WorkerConfig.model_validate(raw)


def test_transfer_engine_transport_validates_control_and_segment_settings(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = {
        "type": "transfer_engine",
        "control_listen": "127.0.0.1:50051",
        "control_advertise": "worker:50051",
        "segment_advertise": "192.0.2.32",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "nvlink_intra",
        "device_name": "auto-discovery",
    }

    config = WorkerConfig.model_validate(raw)

    assert isinstance(config.transport, TransferEngineTransportConfig)
    assert config.transport.max_pending_batches_per_device == 1
    assert config.transport.segment_advertise == "192.0.2.32"
    assert config.transport.protocol == "nvlink_intra"
    assert config.transport.max_workers == 2


def test_transfer_engine_nvlink_requires_cuda(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ggml", device="cpu")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["ggml"] = {"cpu_threads": 1}
    raw["transport"] = {
        "type": "transfer_engine",
        "max_pending_batches_per_device": 1,
        "control_listen": "127.0.0.1:50051",
        "control_advertise": "worker:50051",
        "segment_advertise": "127.0.0.1:12011",
        "protocol": "nvlink_intra",
    }

    with pytest.raises(ValidationError, match="NVLink transports require a CUDA"):
        WorkerConfig.model_validate(raw)

    raw["transport"]["segment_advertise"] = "http://192.0.2.32"  # type: ignore[index]
    raw["transport"]["protocol"] = "tcp"  # type: ignore[index]
    with pytest.raises(ValidationError, match="URL scheme"):
        WorkerConfig.model_validate(raw)


def test_transfer_engine_rdma_requires_explicit_gpu_configuration(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = {
        "type": "transfer_engine",
        "control_listen": "0.0.0.0:52051",
        "control_advertise": "192.0.2.11:52051",
        "segment_advertise": "192.0.2.11:12011",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "rdma",
        "device_name": "mlx5_0",
        "enable_experimental_rdma": True,
    }

    config = WorkerConfig.model_validate(raw)
    assert isinstance(config.transport, TransferEngineTransportConfig)
    assert config.transport.protocol == "rdma"
    assert config.transport.device_name == "mlx5_0"
    assert config.transport.enable_experimental_rdma is True

    raw["transport"]["enable_experimental_rdma"] = False  # type: ignore[index]
    with pytest.raises(ValidationError, match="must be true for RDMA"):
        WorkerConfig.model_validate(raw)

    raw["transport"]["enable_experimental_rdma"] = True  # type: ignore[index]
    raw["transport"]["device_name"] = ""  # type: ignore[index]
    with pytest.raises(ValidationError, match="device_name is required"):
        WorkerConfig.model_validate(raw)

    raw["transport"]["device_name"] = "mlx5_0"  # type: ignore[index]
    raw["transport"]["metadata_server"] = "etcd://192.0.2.12:2379"  # type: ignore[index]
    with pytest.raises(ValidationError, match="must be P2PHANDSHAKE"):
        WorkerConfig.model_validate(raw)


def test_transfer_engine_rdma_requires_a_cuda_worker(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="ggml", device="cpu")
    worker = raw["worker"]
    assert isinstance(worker, dict)
    worker["ggml"] = {"cpu_threads": 1}
    raw["transport"] = {
        "type": "transfer_engine",
        "control_listen": "0.0.0.0:52051",
        "control_advertise": "192.0.2.11:52051",
        "segment_advertise": "192.0.2.11:12011",
        "protocol": "rdma",
        "device_name": "mlx5_0",
        "enable_experimental_rdma": True,
    }

    with pytest.raises(ValidationError, match=r"RDMA transport requires a CUDA Worker"):
        WorkerConfig.model_validate(raw)


def test_transfer_engine_rejects_rdma_gate_for_nvlink(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = {
        "type": "transfer_engine",
        "control_listen": "127.0.0.1:52051",
        "control_advertise": "127.0.0.1:52051",
        "segment_advertise": "127.0.0.1:12011",
        "protocol": "nvlink_intra",
        "enable_experimental_rdma": True,
    }

    with pytest.raises(ValidationError, match="valid only for RDMA"):
        WorkerConfig.model_validate(raw)


@pytest.mark.parametrize(
    "transport",
    [
        {"grpc": {"listen": "127.0.0.1:50051", "advertise": "worker:50051"}},
        {
            "type": "grpc",
            "listen": "127.0.0.1:50051",
            "advertise": "worker:50051",
            "rpc_listen": "127.0.0.1:50052",
        },
        {
            "type": "shm",
            "rpc_listen": "127.0.0.1:50051",
            "rpc_advertise": "worker:50051",
            "listen": "127.0.0.1:50052",
        },
        {
            "type": "nccl",
            "control_listen": "127.0.0.1:50051",
            "control_advertise": "worker:50051",
            "rank": 1,
            "world_size": 2,
            "rendezvous_endpoint": "controller:29500",
            "group_name": "expert-kit",
            "listen": "127.0.0.1:50052",
        },
        {
            "type": "transfer_engine",
            "control_listen": "127.0.0.1:50051",
            "control_advertise": "worker:50051",
            "segment_advertise": "192.0.2.32",
            "listen": "127.0.0.1:50052",
        },
    ],
)
def test_transport_rejects_old_or_mixed_fields(
    tmp_path: Path,
    transport: dict[str, object],
) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["transport"] = transport

    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(raw)


def test_tracing_requires_plaintext_endpoint_and_bounded_sampling(tmp_path: Path) -> None:
    raw = _config(tmp_path, backend="torch", device="cuda:0")
    raw["observability"] = {
        "tracing": {
            "enabled": True,
            "endpoint": "https://collector:4317",
            "sample_ratio": 1,
        }
    }

    with pytest.raises(ValidationError, match="plaintext HTTP endpoint"):
        WorkerConfig.model_validate(raw)

    raw["observability"] = {
        "tracing": {
            "enabled": True,
            "endpoint": "http://collector:4317",
            "sample_ratio": 0,
        }
    }
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkerConfig.model_validate(raw)
