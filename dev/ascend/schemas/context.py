from __future__ import annotations
from typing import Any, Self
from pydantic import BaseModel, computed_field, model_serializer, model_validator
from .cluster import (
    ClusterConfig,
    AttentionConfig,
    ControlConfig,
    ImageConfig,
    Inference,
    ExpertRuntimeConfig,
    PoolConfig,
    NodeConfig,
)
from pathlib import Path
from .experiment import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    ServeConfig,
)


class DuplicateDeviceError(ValueError): ...


class DuplicatePortError(ValueError): ...


class TemplateContext(BaseModel):
    project_name: str
    inference: Inference
    images: ImageConfig
    attention: NodeBindingContext[AttentionConfig]
    control: NodeBindingContext[ControlConfig]
    expert: ExpertContext
    model: ArtifactContext[ModelConfig]
    dataset: ArtifactContext[DatasetConfig]
    serve: ServeConfig
    run: RunConfig
    results_path: Path

    @classmethod
    def from_config(cls, cluster: ClusterConfig, experiment: ExperimentConfig) -> Self:
        nodes = cluster.nodes

        attention = NodeBindingContext[AttentionConfig].resolve(
            config=cluster.attention,
            nodes=nodes,
        )
        control = NodeBindingContext[ControlConfig].resolve(
            config=cluster.control,
            nodes=nodes,
        )
        expert = ExpertContext.resolve(
            pools=cluster.pools,
            nodes=nodes,
            runtime=cluster.expert.runtime,
        )
        paths = cluster.paths

        model = ArtifactContext[ModelConfig].resolve(
            experiment.model,
            paths.models,
            kind="model",
        )
        dataset = ArtifactContext[DatasetConfig].resolve(
            experiment.dataset,
            paths.datasets,
            kind="dataset",
        )
        run = experiment.run

        return cls(
            project_name=cluster.project_name,
            inference=cluster.inference,
            images=cluster.images,
            attention=attention,
            control=control,
            expert=expert,
            model=model,
            dataset=dataset,
            serve=experiment.serve,
            run=run,
            results_path=cluster.paths.results,
        )

    @model_validator(mode="after")
    def validate_device_placement(self) -> Self:
        node = self.attention.config.node
        pool = self.expert.get_pool_by_node(node)

        if pool is None:
            return self

        # Expert and attention are on a same node
        attention_devices = set(self.attention.config.devices)
        pool_devices = set(pool.devices)
        duplicates = attention_devices & pool_devices
        if duplicates:
            raise DuplicateDeviceError(f"Devices {duplicates} have been used.")
        return self

    @model_validator(mode="after")
    def validate_port_placement(self) -> Self:
        def check_duplicate_ports(p1: set[int], p2: set[int]) -> None:
            duplicates = p1 & p2
            if duplicates:
                raise DuplicatePortError(f"Ports {duplicates} have been used.")

        node = self.attention.config.node
        attention_ports = self.attention.config.allocated_ports
        control_ports = self.control.config.allocated_ports
        check_duplicate_ports(attention_ports, control_ports)

        pool = self.expert.get_pool_by_node(node)
        if pool is None:
            return self

        # Expert and attention are on a same node
        pool_ports = pool.allocated_ports
        check_duplicate_ports(attention_ports, pool_ports)
        check_duplicate_ports(control_ports, pool_ports)
        return self

    @computed_field
    @property
    def label(self) -> str:
        num_attention = len(self.attention.config.devices)
        num_workers = self.expert.total_workers
        placement = f"{num_attention}A{num_workers}E"
        dataset_name = self.dataset.config.name

        max_concurrency = self.run.max_concurrency
        output_len = self.run.output_len
        label = f"{placement}-c{max_concurrency}-{dataset_name}-o{output_len}"

        return label


class ExpertContext(BaseModel):
    pools: list[NodeBindingContext[PoolConfig]]
    runtime: ExpertRuntimeConfig

    @property
    def total_workers(self) -> int:
        return sum(pool.config.worker_count for pool in self.pools)

    @classmethod
    def resolve(
        cls,
        pools: list[PoolConfig],
        nodes: dict[str, NodeConfig],
        runtime: ExpertRuntimeConfig,
    ) -> Self:
        resolved_pools = [
            NodeBindingContext[PoolConfig].resolve(
                config=pool,
                nodes=nodes,
            )
            for pool in pools
        ]

        return cls(pools=resolved_pools, runtime=runtime)

    def get_pool_by_node(self, node: str) -> PoolConfig | None:
        for pool in self.pools:
            if pool.config.node != node:
                continue
            return pool.config
        return None


class NodeBindingContext[T: AttentionConfig | ControlConfig | PoolConfig](BaseModel):
    node: NodeConfig
    config: T

    @classmethod
    def resolve(
        cls,
        config: T,
        nodes: dict[str, NodeConfig],
    ) -> Self:
        node = resolve_ref(config.node, nodes, kind="node")
        return cls(node=node, config=config)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            **self.config.model_dump(mode="json"),
            **self.node.model_dump(mode="json"),
        }


class ArtifactContext[T: ModelConfig | DatasetConfig](BaseModel):
    config: T
    path: Path | None = None

    @classmethod
    def resolve(
        cls,
        config: T,
        paths: dict[str, Path],
        *,
        kind: str,
    ) -> Self:
        path_ref = config.path_ref
        if path_ref is None:
            return cls(config=config)

        path = resolve_ref(path_ref, paths, kind=kind)
        return cls(path=path, config=config)

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return {
            **self.config.model_dump(mode="json"),
            "path": self.path,
        }


def resolve_ref[T](ref: str, registry: dict[str, T], *, kind: str) -> T:
    if ref not in registry:
        raise KeyError(f"{kind} ref {ref} not found in registered list.")

    return registry[ref]
