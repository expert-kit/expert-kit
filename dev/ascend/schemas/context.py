from __future__ import annotations
from typing import Any, Self
from pydantic import BaseModel, computed_field, model_serializer
from .cluster import (
    ClusterConfig,
    AttentionConfig,
    WorkerConfig,
    ControlConfig,
    ImageConfig,
    Inference,
    ExpertConfig,
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


class GenerationContext(BaseModel):
    project_name: str
    inference: Inference
    images: ImageConfig
    attention: RoleContext[AttentionConfig]
    control: RoleContext[ControlConfig]
    expert: RoleContext[ExpertConfig]
    model: ArtifactContext[ModelConfig]
    dataset: ArtifactContext[DatasetConfig]
    serve: ServeConfig
    run: RunConfig
    workers: list[WorkerConfig]
    results_path: Path

    @classmethod
    def from_config(cls, cluster: ClusterConfig, experiment: ExperimentConfig) -> Self:
        nodes = cluster.nodes

        attention = RoleContext[AttentionConfig].resolve(
            config=cluster.roles.attention,
            nodes=nodes,
        )
        control = RoleContext[ControlConfig].resolve(
            config=cluster.roles.control,
            nodes=nodes,
        )
        expert = RoleContext[ExpertConfig].resolve(
            config=cluster.roles.expert,
            nodes=nodes,
        )
        workers = cluster.roles.expert.workers.generate_workers()
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
            workers=workers,
            results_path=cluster.paths.results,
        )

    @computed_field
    @property
    def label(self) -> str:
        num_attention = len(self.attention.config.devices)
        num_workers = len(self.workers)
        placement = f"{num_attention}A{num_workers}E"
        dataset_name = self.dataset.config.name

        max_concurrency = self.run.max_concurrency
        output_len = self.run.output_len
        label = f"{placement}-c{max_concurrency}-{dataset_name}-o{output_len}"

        return label


class RoleContext[T: AttentionConfig | ControlConfig | ExpertConfig](BaseModel):
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
