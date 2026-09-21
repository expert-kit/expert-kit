"""Typer entry point for the shared Torch model benchmark."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from expertkit_torch.benchmark.config import (
    BenchmarkConfig,
    BenchmarkDtype,
    BenchmarkMode,
    DatasetName,
    DevicePlatform,
    LauncherKind,
)
from expertkit_torch.benchmark.launcher import GlobalBenchmarkReport, select_launcher

app = typer.Typer(
    help="Run Expert Kit Torch frontend benchmarks.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _configure(
    context: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Generated Torch benchmark YAML used as run defaults.",
        ),
    ] = None,
) -> None:
    """Load an optional config before the run command parses its options."""

    if config is not None:
        loaded = BenchmarkConfig.from_yaml(config)
        context.default_map = {
            "run": loaded.model_dump(mode="python", by_alias=False),
        }


@app.command("run")
def run(
    model_path: Annotated[Path, typer.Option(help="Host model checkpoint directory.")],
    dataset_path: Annotated[Path, typer.Option(help="Host ShareGPT JSON path.")],
    device_platform: Annotated[
        DevicePlatform,
        typer.Option(help="Torch device platform."),
    ],
    device_ids: Annotated[
        list[int],
        typer.Option(help="One or more device IDs, in rank order."),
    ],
    num_prompts: Annotated[int, typer.Option(min=1)],
    max_concurrency: Annotated[int, typer.Option(min=1)],
    output_length: Annotated[int, typer.Option(min=1)],
    mode: BenchmarkMode = BenchmarkMode.EXPERTKIT,
    controller_endpoint: str | None = None,
    instance_id: int | None = None,
    launcher: LauncherKind = LauncherKind.AUTO,
    dtype: BenchmarkDtype = BenchmarkDtype.AUTO,
    dataset_name: DatasetName = DatasetName.SHAREGPT,
    seed: int = 0,
    warmup_runs: Annotated[int, typer.Option(min=0)] = 1,
    json_output: Path | None = None,
) -> None:
    """Run one inline or multi-device Torch benchmark."""

    config = BenchmarkConfig(
        mode=mode,
        controller_endpoint=controller_endpoint,
        instance_id=instance_id,
        launcher=launcher,
        device_platform=device_platform,
        device_ids=tuple(device_ids),
        model_path=model_path,
        dtype=dtype,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        seed=seed,
        num_prompts=num_prompts,
        max_concurrency=max_concurrency,
        output_length=output_length,
        warmup_runs=warmup_runs,
        json_output=json_output,
    ).resolve_paths(Path.cwd())

    report = select_launcher(config).launch(config)
    typer.echo(format_report(report))
    if config.json_output is not None:
        config.json_output.parent.mkdir(parents=True, exist_ok=True)
        config.json_output.write_text(
            json.dumps(
                report.as_dict(configuration=config.model_dump(mode="json", by_alias=False)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def format_report(report: GlobalBenchmarkReport) -> str:
    """Format global and per-rank throughput results."""

    lines = [
        f"Model: {report.model_type}  Mode: {report.mode}",
        f"Ranks: {len(report.rank_reports)}  Prompts: {report.num_prompts}",
        f"Benchmark duration (s): {report.duration_seconds:.2f}",
        f"Input tokens: {report.input_tokens}",
        f"Output tokens: {report.output_tokens}",
        f"Request throughput (req/s): {report.request_throughput:.2f}",
        f"Output throughput (tok/s): {report.output_throughput:.2f}",
        f"Total throughput (tok/s): {report.total_throughput:.2f}",
        "Rank  Device  Prompts  Batch/rank  Duration (s)",
    ]
    lines.extend(
        f"{rank.assignment.rank:>4}  "
        f"{rank.assignment.device_id:>6}  "
        f"{rank.benchmark.num_prompts:>7}  "
        f"{rank.benchmark.batches[0].batch_size:>10}  "
        f"{rank.duration:>12.2f}"
        for rank in report.rank_reports
    )
    return "\n".join(lines)


def main(args: Sequence[str] | None = None) -> None:
    """Console-script entry point."""

    app(args=args)


if __name__ == "__main__":
    main()
