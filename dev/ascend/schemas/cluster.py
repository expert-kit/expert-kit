"""Cluster, node, pool, and path configuration models."""

from __future__ import annotations

from ipaddress import IPv4Address
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, computed_field, model_validator

from .config import ConfigModel


class ClusterConfig(ConfigModel):
    """Complete deployment topology and artifact path configuration."""

    project_name: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    inference: Inference
    images: ImageConfig
    attention: AttentionConfig
    control: ControlConfig
    expert: ExpertConfig
    pools: list[PoolConfig]
    nodes: dict[str, NodeConfig]
    paths: PathConfig
    transport: AscendTransportConfig = Field(default_factory=lambda: AscendTransportConfig())

    @classmethod
    def from_yaml(cls, file: Path) -> Self:
        data = yaml.safe_load(file.read_text())
        return cls.model_validate(data)

    @model_validator(mode="after")
    def validate_unique_pool_ids(self) -> Self:
        allocated_ids = set()
        for pool in self.pools:
            if pool.id in allocated_ids:
                raise ValueError(f"Pool IDs must be unique: {pool.id}")
            allocated_ids.add(pool.id)
        return self


class Inference(ConfigModel):
    """Controller instance naming configuration."""

    instance_name: str


class ImageConfig(ConfigModel):
    """Container images used by generated services."""

    rust: str
    uv: str
    control_base: str
    vllm_ascend: str
    postgres: str
    control: str
    attention: str
    expert: str


class NodeConfig(ConfigModel):
    """Addressable host in the deployment topology."""

    address: IPv4Address


class AttentionConfig(ConfigModel):
    """Frontend node, port, and device placement."""

    node: str
    vllm_port: int
    devices: list[int]

    @computed_field
    @property
    def visible_devices(self) -> str:
        return ",".join(str(device) for device in self.devices)

    @computed_field
    @property
    def dp_size(self) -> int:
        return len(self.devices)

    @property
    def allocated_ports(self) -> set[int]:
        return {
            self.vllm_port,
        }


class ControlConfig(ConfigModel):
    """Controller, database, and control-plane port configuration."""

    node: str
    postgres_port: int
    worker_control_port: int
    frontend_control_port: int
    weight_server_port: int
    database: DatabaseConfig
    fault_detection: FaultDetectionConfig

    @property
    def allocated_ports(self) -> set[int]:
        return {
            self.postgres_port,
            self.worker_control_port,
            self.frontend_control_port,
            self.weight_server_port,
        }


class DatabaseConfig(ConfigModel):
    """PostgreSQL credentials and connection limits."""

    user: str
    password: str
    name: str
    max_connections: int


class FaultDetectionConfig(ConfigModel):
    """Heartbeat and node failure detection thresholds."""

    heartbeat_timeout_secs: int
    node_active_threshold_secs: int
    poller_interval_secs: int


class ExpertConfig(ConfigModel):
    """Expert worker runtime configuration."""

    runtime: ExpertRuntimeConfig


class AscendTransportConfig(ConfigModel):
    """Select the A/E data backend without changing the model adapter."""

    type: Literal["grpc", "transfer_engine"] = "grpc"
    protocol: Literal["ascend_direct"] = "ascend_direct"
    max_workers: int = Field(default=2, gt=0)
    client_max_in_flight: int = Field(default=1, gt=0)
    max_registered_bytes: int | None = Field(default=None, gt=0)


class ExpertRuntimeConfig(ConfigModel):
    """Resource and lifecycle limits shared by expert workers."""

    max_batch_tokens: int
    max_active_batches_per_device: int
    device_memory_limit: str
    shutdown_grace_secs: int
    max_pending_batches_per_device: int
    heartbeat_interval_secs: int
    heartbeat_timeout_secs: int
    max_concurrent_loads: int


class PoolConfig(ConfigModel):
    """A group of workers placed on one node."""

    id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    node: str
    devices: list[int]
    start_worker_port: int
    start_peer_port: int

    @property
    def worker_count(self) -> int:
        return len(self.devices)

    @computed_field
    @property
    def workers(self) -> list[WorkerConfig]:
        workers: list[WorkerConfig] = []
        for idx, device in enumerate(self.devices):
            workers.append(
                WorkerConfig(
                    idx=idx,
                    device=device,
                    port=self.start_worker_port + idx,
                    peer_port=self.start_peer_port + idx,
                )
            )
        return workers

    @property
    def allocated_ports(self) -> set[int]:
        ports = set()
        for worker in self.workers:
            ports.add(worker.port)
            ports.add(worker.peer_port)

        return ports


class WorkerConfig(ConfigModel):
    """One generated worker's device and listening ports."""

    idx: int
    device: int
    port: int
    peer_port: int

    @computed_field
    @property
    def id(self) -> str:
        return f"worker-{self.idx:02d}"

    @computed_field
    @property
    def device_name(self) -> str:
        return f"npu:{self.device}"


class PathConfig(ConfigModel):
    """Host paths mounted into generated services."""

    models: dict[str, Path]
    datasets: dict[str, Path]
    results: Path
