from __future__ import annotations
import yaml
from typing import Self
from pydantic import Field, IPvAnyAddress, computed_field
from pathlib import Path
from .config import ConfigModel


class ClusterConfig(ConfigModel):
    project_name: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    inference: Inference
    images: ImageConfig
    roles: RolesConfig
    nodes: dict[str, NodeConfig]
    paths: PathConfig

    @classmethod
    def from_yaml(cls, file: Path) -> Self:
        data = yaml.safe_load(file.read_text())
        return cls.model_validate(data)


class Inference(ConfigModel):
    instance_name: str


class ImageConfig(ConfigModel):
    postgres: str
    control: str
    attention: str
    expert: str


class RolesConfig(ConfigModel):
    attention: AttentionConfig
    control: ControlConfig
    expert: ExpertConfig


class NodeConfig(ConfigModel):
    address: IPvAnyAddress


class AttentionConfig(ConfigModel):
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


class ControlConfig(ConfigModel):
    node: str
    postgres_port: int
    worker_control_port: int
    frontend_control_port: int
    weight_server_port: int
    database: DatabaseConfig
    fault_detection: FaultDetectionConfig


class DatabaseConfig(ConfigModel):
    user: str
    password: str
    name: str
    max_connections: int


class FaultDetectionConfig(ConfigModel):
    heartbeat_timeout_secs: int
    node_active_threshold_secs: int
    poller_interval_secs: int


class ExpertConfig(ConfigModel):
    node: str
    workers: WorkersMeta


class WorkersMeta(ConfigModel):
    devices: list[int]
    start_worker_port: int
    start_peer_port: int
    runtime: WorkerRuntimeConfig

    def generate_workers(self) -> list[WorkerConfig]:
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


class WorkerRuntimeConfig(ConfigModel):
    max_batch_tokens: int
    max_active_batches_per_device: int
    device_memory_limit: str
    shutdown_grace_secs: int
    max_pending_batches_per_device: int
    heartbeat_interval_secs: int
    heartbeat_timeout_secs: int
    max_concurrent_loads: int


class WorkerConfig(ConfigModel):
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
    models: dict[str, Path]
    datasets: dict[str, Path]
    results: Path
