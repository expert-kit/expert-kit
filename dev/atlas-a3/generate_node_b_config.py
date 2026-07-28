from __future__ import annotations
from expertkit_worker import WorkerConfig
from expertkit_worker.config.models import (
    ModelConfig,
    LoggingConfig,
    ObservabilityConfig,
)
import yaml
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from models.templates import (
    WorkerTemplate,
    TransportTemplate,
    ControllerTemplate,
    WeightManagerTemplate,
    DeploymentEnv,
    WorkerDeployment,
)

from models.compose import NodeBCompose, WorkerCompose


DEPLOYMENT_ROOT = Path(__file__).resolve().parent

INVENTORY_YAML = DEPLOYMENT_ROOT / "node-b.inventory.yaml"
GENERATED_ROOT = DEPLOYMENT_ROOT / "generated"
WORKER_CONF_ROOT = GENERATED_ROOT / "workers"
COMPOSE_YAML = GENERATED_ROOT / "compose.node-b.yaml"


class WorkerInventory(BaseModel):
    worker_count: int = Field(gt=0)
    device_count: int = Field(default=16, gt=0)
    worker_base_port: int = 51234
    peer_base_port: int = 52234

    model: ModelConfig
    worker: WorkerTemplate
    transport: TransportTemplate
    controller: ControllerTemplate
    weight_manager: WeightManagerTemplate
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @model_validator(mode="after")
    def validate_device_range(self) -> WorkerInventory:
        start = self.worker.device_start_index
        stop = start + self.worker_count

        if stop > self.device_count:
            raise ValueError(
                f"worker device range [{start}, {stop}) exceeds "
                f"available device range [0, {self.device_count})"
            )

        return self


def read_inventory(inventory_file: Path) -> WorkerInventory:
    with inventory_file.open("r") as f:
        config = yaml.safe_load(f)
    return WorkerInventory.model_validate(config)


def create_worker_config(
    inventory: WorkerInventory,
    deploy: WorkerDeployment,
    env: DeploymentEnv,
) -> WorkerConfig:
    return WorkerConfig(
        model=inventory.model,
        worker=inventory.worker.materialize(deploy),
        transport=inventory.transport.materialize(deploy),
        controller=inventory.controller.materialize(env),
        weight_manager=inventory.weight_manager.materialize(env, deploy),
        logging=inventory.logging,
        observability=inventory.observability,
    )


def save_config(
    model: BaseModel,
    file: Path,
    exclude_none: bool = True,
) -> None:
    blocks = []
    fields = model.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=exclude_none,
    )
    for field, value in fields.items():
        blocks.append(
            yaml.safe_dump(
                {field: value},
                sort_keys=False,
            )
        )

    # Add lines between fields
    file.write_text("\n".join(blocks))


def main() -> None:
    # 2. After path is confirmed, set WorkerInstance and convert
    # it to docker compose service

    WORKER_CONF_ROOT.mkdir(parents=True, exist_ok=True)
    inventory = read_inventory(INVENTORY_YAML)
    env = DeploymentEnv()  # ty: ignore[missing-argument]

    node_b_compose = NodeBCompose()

    for idx in range(inventory.worker_count):
        # Gather current worker deployment info
        deploy = WorkerDeployment(
            worker_id=f"worker-{idx:02d}",
            host_ip=env.node_b_ip,
            device_index=inventory.worker.device_start_index + idx,
            worker_host_port=env.worker_base_port + idx,
            peer_host_port=env.peer_base_port + idx,
            worker_container_port=inventory.worker_base_port + idx,
            peer_container_port=inventory.peer_base_port + idx,
        )

        # Create worker config from inventory.yaml and deployment ENV
        worker = create_worker_config(inventory, deploy, env)

        # Save configs to workers/
        worker_yaml = WORKER_CONF_ROOT / f"{deploy.worker_id}.yaml"
        save_config(worker, worker_yaml)
        print(f"Worker config saved to {worker_yaml}!")

        # Compose fields for current worker
        worker_id = deploy.worker_id
        node_b_compose.services[worker_id] = WorkerCompose.from_deployment(deploy)

    save_config(node_b_compose, COMPOSE_YAML)
    print(f"Node B Compose saved to {COMPOSE_YAML}!")


if __name__ == "__main__":
    main()
