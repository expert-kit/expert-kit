from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path


from renderer import Renderer
from schemas.cluster import WorkerConfig, PoolConfig
from schemas.context import TemplateContext, NodeBindingContext
from schemas.experiment import DatasetType
from utils import long_banner

ASCEND_DIR = Path(__file__).resolve().parent
DEFAULT_CLUSTER_CONFIG = ASCEND_DIR / "configs" / "cluster.yaml"
DEFAULT_EXPERIMENT_CONFIG = ASCEND_DIR / "configs" / "experiment.yaml"
TEMPLATE_DIR = ASCEND_DIR / "templates"
DEFAULT_OUTPUT_DIR = ASCEND_DIR / "generated"


class ConfigFileNames:
    COMPOSE_BUILD = "compose.build.yaml"
    COMPOSE_ATTENTION_DEV = "compose.attention.dev.yaml"
    COMPOSE_ATTENTION = "compose.attention.yaml"
    COMPOSE_EXPERT_DEV = "compose.expert.dev.yaml"
    COMPOSE_EXPERT = "compose.expert.yaml"
    VLLM_SERVE = "vllm-serve.yaml"
    VLLM_BENCH = "vllm-bench.yaml"
    TORCH_BENCH = "torch-bench.yaml"
    CONTROLLER = "controller.yaml"
    WORKER = "worker.yaml"


@dataclass(frozen=True, slots=True)
class RenderSpec:
    output: Path
    template: str


def generate_root_config(
    renderer: Renderer,
    context: TemplateContext,
    output: Path,
) -> None:
    long_banner("Generating singleton YAML")
    output.mkdir(exist_ok=True, parents=True)
    template_context = context.model_dump(mode="json")
    for spec in root_render_specs(output):
        renderer.render_to_file(
            spec.output,
            spec.template,
            **template_context,
        )

    if context.dataset.config.type is DatasetType.SHAREGPT:
        spec = create_render_spec(output, ConfigFileNames.TORCH_BENCH)
        renderer.render_to_file(
            spec.output,
            spec.template,
            **template_context,
        )


def generate_pools_config(
    renderer: Renderer,
    context: TemplateContext,
    output: Path,
) -> None:
    def generate_pool_config(pool: NodeBindingContext[PoolConfig]) -> None:
        pool_id = pool.config.id
        pool_dir = output / pool_id
        long_banner(f"Generating pool {pool_id} YAML")

        generate_expert_compose_config(
            renderer=renderer,
            pool=pool,
            context=context,
            output=pool_dir,
        )
        generate_workers_config(
            renderer=renderer,
            pool=pool,
            context=context,
            output=pool_dir / "workers",
        )

    long_banner("Generating pools YAML")
    output.mkdir(exist_ok=True, parents=True)
    for pool in context.expert.pools:
        generate_pool_config(pool)


def generate_expert_compose_config(
    renderer: Renderer,
    pool: NodeBindingContext[PoolConfig],
    context: TemplateContext,
    output: Path,
) -> None:
    long_banner("Generating expert compose YAML")
    template_context = context.model_dump(mode="json")
    output.mkdir(exist_ok=True, parents=True)
    for spec in expert_compose_render_specs(output):
        renderer.render_to_file(
            spec.output,
            spec.template,
            pool=pool.model_dump(mode="json"),
            **template_context,
        )


def generate_workers_config(
    renderer: Renderer,
    pool: NodeBindingContext[PoolConfig],
    context: TemplateContext,
    output: Path,
) -> None:
    long_banner("Generating workers YAML")
    template_context = context.model_dump(mode="json")

    def generate_worker_config(worker: WorkerConfig) -> None:
        renderer.render_to_file(
            output / f"{worker.id}.yaml",
            f"{ConfigFileNames.WORKER}.jinja",
            worker=worker.model_dump(mode="json"),
            pool=pool.model_dump(mode="json"),
            **template_context,
        )

    long_banner("Generating worker YAML")
    output.mkdir(exist_ok=True, parents=True)
    for worker in pool.config.workers:
        generate_worker_config(worker)


def create_render_spec(output: Path, file_name: str) -> RenderSpec:
    return RenderSpec(
        output=output / file_name,
        template=f"{file_name}.jinja",
    )


def root_render_specs(output: Path) -> tuple[RenderSpec, ...]:
    return (
        create_render_spec(output, ConfigFileNames.COMPOSE_BUILD),
        create_render_spec(output, ConfigFileNames.COMPOSE_ATTENTION),
        create_render_spec(output, ConfigFileNames.COMPOSE_ATTENTION_DEV),
        create_render_spec(output, ConfigFileNames.CONTROLLER),
        create_render_spec(output, ConfigFileNames.VLLM_BENCH),
        create_render_spec(output, ConfigFileNames.VLLM_SERVE),
    )


def expert_compose_render_specs(output: Path) -> tuple[RenderSpec, ...]:
    return (
        create_render_spec(output, ConfigFileNames.COMPOSE_EXPERT),
        create_render_spec(output, ConfigFileNames.COMPOSE_EXPERT_DEV),
    )
