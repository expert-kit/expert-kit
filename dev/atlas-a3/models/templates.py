from __future__ import annotations
from dataclasses import dataclass
from expertkit_worker.config.models import (
    BackendName,
    PositiveByteSize,
    GgmlConfig,
    DiskCacheConfig,
    WorkerProcessConfig,
    ControllerConfig,
    GrpcTransportConfig,
    ShmTransportConfig,
    WeightManagerConfig,
    PeerWeightConfig,
    AbsolutePath,
)


from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    AnyHttpUrl,
)
from pydantic import IPvAnyAddress
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkerDeployment:
    worker_id: str
    host_ip: IPvAnyAddress
    device_index: int
    worker_container_port: int
    worker_host_port: int
    peer_container_port: int
    peer_host_port: int


class DeploymentEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.with_name(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    controller_port: int = Field(validation_alias="EK_CONTROLLER_INTRA_PORT")
    weight_server_port: int = Field(validation_alias="EK_WEIGHT_SERVER_PORT")

    node_a_ip: IPvAnyAddress = Field(validation_alias="EK_NODE_A_IP")
    node_b_ip: IPvAnyAddress = Field(validation_alias="EK_NODE_B_IP")

    worker_base_port: int = Field(validation_alias="EK_WORKER_BASE_PORT")
    peer_base_port: int = Field(validation_alias="EK_PEER_BASE_PORT")

    @property
    def controller_endpoint(self) -> str:
        return f"{self.node_a_ip}:{self.controller_port}"

    @property
    def weight_server_endpoint(self) -> AnyHttpUrl:
        return AnyHttpUrl(f"http://{self.node_a_ip}:{self.weight_server_port}")


class _Template(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


# Modified based on ek-worker/src/expertkit_worker/config/models.py
# Use `materialize` to convert template to the real config


class WorkerTemplate(_Template):
    backend: BackendName = BackendName.TORCH
    device_start_index: int = Field(ge=0, exclude=True)
    device_prefix: str = Field(exclude=True)
    max_batch_tokens: int = Field(default=4096, gt=0)
    max_active_batches_per_device: int = Field(default=1, gt=0)
    device_memory_limit: PositiveByteSize
    shutdown_grace_secs: float = Field(default=30.0, gt=0)
    ggml: GgmlConfig | None = None

    def materialize(self, deploy: WorkerDeployment) -> WorkerProcessConfig:
        return WorkerProcessConfig(
            id=deploy.worker_id,
            **self.model_dump(),
            device=f"{self.device_prefix}:{deploy.device_index}",
        )


class GrpcTransportTemplate(_Template):
    type: Literal["grpc"]
    max_pending_batches_per_device: int = Field(gt=0)

    def listen(self, deploy: WorkerDeployment) -> str:
        return f"0.0.0.0:{deploy.worker_container_port}"

    def advertise(self, deploy: WorkerDeployment) -> str:
        return f"{deploy.host_ip}:{deploy.worker_host_port}"

    def materialize(self, deploy: WorkerDeployment) -> GrpcTransportConfig:
        return GrpcTransportConfig(
            **self.model_dump(),
            listen=self.listen(deploy),
            advertise=self.advertise(deploy),
        )


class ShmTransportTemplate(_Template):
    type: Literal["shm"]
    max_pending_batches_per_device: int = Field(gt=0)
    shared_memory_dir: Literal["/dev/shm"] = "/dev/shm"

    def listen(self, deploy: WorkerDeployment) -> str:
        return f"0.0.0.0:{deploy.worker_container_port}"

    def advertise(self, deploy: WorkerDeployment) -> str:
        # NOTE: This is a host ip, thus a host port is needed but not container port
        return f"{deploy.host_ip}:{deploy.worker_host_port}"

    def materialize(self, deploy: WorkerDeployment) -> ShmTransportConfig:
        return ShmTransportConfig(
            **self.model_dump(),
            rpc_listen=self.listen(deploy),
            rpc_advertise=self.advertise(deploy),
        )


TransportTemplate = Annotated[
    GrpcTransportTemplate | ShmTransportTemplate,
    Field(discriminator="type"),
]


class ControllerTemplate(_Template):
    heartbeat_interval_secs: float = Field(default=3.0, gt=0)
    heartbeat_timeout_secs: float = Field(default=10.0, gt=0)

    def materialize(self, env: DeploymentEnv) -> ControllerConfig:
        return ControllerConfig(
            endpoint=env.controller_endpoint,
            **self.model_dump(),
        )


class DiskCacheTemplate(_Template):
    base_path: AbsolutePath
    writeback: bool

    def materialize(self, deploy: WorkerDeployment) -> DiskCacheConfig:
        return DiskCacheConfig(
            path=self.base_path / deploy.worker_id,
            writeback=self.writeback,
        )


class WeightManagerTemplate(_Template):
    max_concurrent_loads: int = Field(default=64, gt=0)
    disk_cache: DiskCacheTemplate

    def materialize(
        self,
        env: DeploymentEnv,
        deploy: WorkerDeployment,
    ) -> WeightManagerConfig:
        return WeightManagerConfig(
            max_concurrent_loads=self.max_concurrent_loads,
            disk_cache=self.disk_cache.materialize(deploy),
            peer=PeerWeightConfig(
                listen=self.peer_listen(deploy),
                advertise=self.peer_advertise(deploy),
            ),
            weight_server_endpoint=env.weight_server_endpoint,
        )

    def peer_listen(self, deploy: WorkerDeployment) -> str:
        return f"0.0.0.0:{deploy.peer_container_port}"

    def peer_advertise(self, deploy: WorkerDeployment) -> AnyHttpUrl:
        # NOTE: This is a host ip, thus a host port is needed but not container port
        return AnyHttpUrl(f"http://{deploy.host_ip}:{deploy.peer_host_port}")
