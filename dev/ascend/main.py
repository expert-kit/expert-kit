from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from renderer import Renderer
from schemas.cluster import ClusterConfig, WorkerConfig
from schemas.context import GenerationContext
from schemas.experiment import ExperimentConfig
from utils import long_banner

ASCEND_DIR = Path(__file__).resolve().parent
DEFAULT_CLUSTER_CONFIG = ASCEND_DIR / "configs" / "cluster.yaml"
DEFAULT_EXPERIMENT_CONFIG = ASCEND_DIR / "configs" / "experiment.yaml"
DEFAULT_OUTPUT_DIR = ASCEND_DIR / "generated"

app = typer.Typer(
    help="Validate and render Ascend deployment configuration.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main() -> None:
    """Validate and render Ascend deployment configuration."""


class Templates:
    DIR = ASCEND_DIR / "templates"

    COMPOSE_ATTENTION_DEV = "compose.attention.dev.yaml.jinja"
    COMPOSE_ATTENTION = "compose.attention.yaml.jinja"
    COMPOSE_EXPERT_DEV = "compose.expert.dev.yaml.jinja"
    COMPOSE_EXPERT = "compose.expert.yaml.jinja"
    VLLM_SERVE = "vllm-serve.yaml.jinja"
    VLLM_BENCH = "vllm-bench.yaml.jinja"
    CONTROLLER = "controller.yaml.jinja"
    WORKER = "worker.yaml.jinja"


@dataclass(frozen=True, slots=True)
class GeneratedPaths:
    root: Path

    @property
    def workers(self) -> Path:
        return self.root / "workers"

    def singleton_outputs(self) -> tuple[tuple[Path, str], ...]:
        return (
            (self.root / "compose.attention.dev.yaml", Templates.COMPOSE_ATTENTION_DEV),
            (self.root / "compose.attention.yaml", Templates.COMPOSE_ATTENTION),
            (self.root / "compose.expert.dev.yaml", Templates.COMPOSE_EXPERT_DEV),
            (self.root / "compose.expert.yaml", Templates.COMPOSE_EXPERT),
            (self.root / "controller.yaml", Templates.CONTROLLER),
            (self.root / "vllm-serve.yaml", Templates.VLLM_SERVE),
            (self.root / "vllm-bench.yaml", Templates.VLLM_BENCH),
        )


def generate_singleton_yamls(
    renderer: Renderer,
    context: GenerationContext,
    paths: GeneratedPaths,
) -> None:
    long_banner("Generating singleton YAML")
    template_context = context.model_dump(mode="json")
    for output_file, template_name in paths.singleton_outputs():
        renderer.render_to_file(output_file, template_name, **template_context)


def generate_worker_yamls(
    renderer: Renderer,
    context: GenerationContext,
    paths: GeneratedPaths,
) -> None:
    long_banner("Generating Worker YAML")
    template_context = context.model_dump(mode="json")

    def generate_worker_yaml(worker: WorkerConfig) -> None:
        renderer.render_to_file(
            paths.workers / f"{worker.id}.yaml",
            Templates.WORKER,
            worker=worker,
            **template_context,
        )

    for worker in context.workers:
        generate_worker_yaml(worker)


@app.command()
def generate(
    cluster: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Cluster YAML.",
        ),
    ] = DEFAULT_CLUSTER_CONFIG,
    experiment: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Experiment YAML.",
        ),
    ] = DEFAULT_EXPERIMENT_CONFIG,
    output: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            help="Generated output directory.",
        ),
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Generate role-specific Compose and runtime YAML."""
    cluster_config = ClusterConfig.from_yaml(cluster)
    experiment_config = ExperimentConfig.from_yaml(experiment)
    context = GenerationContext.from_config(cluster_config, experiment_config)

    paths = GeneratedPaths(output)
    paths.workers.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(Templates.DIR)
    generate_singleton_yamls(renderer, context, paths)
    generate_worker_yamls(renderer, context, paths)


if __name__ == "__main__":
    app()
