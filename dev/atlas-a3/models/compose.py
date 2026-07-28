from __future__ import annotations
from typing import Literal
from pydantic import (
    BaseModel,
    IPvAnyAddress,
    Field,
)

from .templates import WorkerDeployment


def default_networks() -> dict[str, ComposeNetwork]:
    return {"expert-kit-network": ComposeNetwork()}


def default_volumes() -> dict[str, dict]:
    return {"worker-cache": {}}


def default_bind() -> dict[str, bool]:
    return {"create_host_path": False}


class NodeBCompose(BaseModel):
    services: dict[str, WorkerCompose] = Field(default_factory=dict)
    networks: dict[str, ComposeNetwork] = Field(default_factory=default_networks)
    volumes: dict[str, dict] = Field(default_factory=default_volumes)


class ComposeNetwork(BaseModel):
    driver: str = "bridge"


class WorkerCompose(BaseModel):
    extends: ComposeExtends
    volumes: list[ComposeVolume]
    ports: list[ComposePort]

    @classmethod
    def from_deployment(cls, deploy: WorkerDeployment) -> WorkerCompose:
        extends = ComposeExtends()
        volumes = [ComposeVolume(source=f"./workers/{deploy.worker_id}.yaml")]
        ports = [
            # Worker
            ComposePort(
                target=deploy.worker_container_port,
                published=str(deploy.worker_host_port),
                host_ip=deploy.host_ip,
            ),
            # Peer
            ComposePort(
                target=deploy.peer_container_port,
                published=str(deploy.peer_host_port),
                host_ip=deploy.host_ip,
            ),
        ]

        return cls(
            extends=extends,
            volumes=volumes,
            ports=ports,
        )


class ComposeExtends(BaseModel):
    file: str = "../compose.worker-common.yaml"
    service: str = "worker-base"


class ComposeVolume(BaseModel):
    source: str
    type: str = "bind"
    target: str = "/etc/expert-kit/worker.yaml"
    read_only: bool = True
    bind: dict[str, bool] = Field(default_factory=default_bind)


class ComposePort(BaseModel):
    target: int
    host_ip: IPvAnyAddress
    published: str
    protocol: Literal["tcp", "udp"] = "tcp"
