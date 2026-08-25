"""Command-line entry point for the shared Torch model benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from expertkit_torch.benchmark.datasets import ShareGPTDataset
from expertkit_torch.benchmark.runner import BenchmarkReport, run_benchmark
from expertkit_torch.models import load_model


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ek-torch-benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("expertkit", "local"), default="expertkit")
    parser.add_argument("--controller-endpoint", default="127.0.0.1:5002")
    parser.add_argument("--instance-id", type=_positive_int)
    parser.add_argument("--batch-sizes", nargs="+", type=_positive_int, default=[1])
    parser.add_argument("--dataset-name", choices=("sharegpt",), default="sharegpt")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=_positive_int, default=16)
    parser.add_argument("--output-length", type=_positive_int, default=20)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def _format_optional(value: object, format_spec: str) -> str:
    if value is None:
        return "-"
    return format(value, format_spec)


def format_report(report: BenchmarkReport) -> str:
    """Format median benchmark results as a compact table."""

    lines = [
        f"Model: {report.model_type}  Mode: {report.mode}",
        (
            "Batch  Input tok  Pad len  Output  Prefill ms  Prefill tok/s  "
            "Decode ms  Decode tok/s  Decode ms/step  Total ms  Output tok/s"
        ),
    ]
    for batch in report.batches:
        result = batch.median()
        lines.append(
            f"{result['batch_size']:>5}  "
            f"{result['input_tokens']:>9.0f}  "
            f"{result['padded_input_length']:>7.0f}  "
            f"{result['output_length']:>6}  "
            f"{result['prefill_seconds'] * 1000:>10.2f}  "
            f"{result['prefill_tps']:>13.2f}  "
            f"{result['decode_seconds'] * 1000:>9.2f}  "
            f"{_format_optional(result['decode_tps'], '>12.2f')}  "
            f"{_format_optional(result['decode_step_ms'], '>14.2f')}  "
            f"{result['total_seconds'] * 1000:>8.2f}  "
            f"{result['output_tps']:>12.2f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Load a supported model, run the benchmark, and print median results."""

    arguments = _parser().parse_args(argv)
    dataset = ShareGPTDataset(arguments.dataset_path, seed=arguments.seed)
    with load_model(
        arguments.model_path,
        mode=arguments.mode,
        controller_endpoint=arguments.controller_endpoint,
        instance_id=arguments.instance_id,
        device=arguments.device,
        dtype=arguments.dtype,
    ) as loaded:
        report = run_benchmark(
            loaded.model,
            loaded.tokenizer,
            dataset,
            model_type=loaded.model_type,
            mode=arguments.mode,
            batch_sizes=arguments.batch_sizes,
            num_prompts=arguments.num_prompts,
            output_length=arguments.output_length,
            warmup_runs=arguments.warmup_runs,
            device=arguments.device,
        )
    print(format_report(report))
    if arguments.json_output is not None:
        payload = report.as_dict()
        payload["configuration"] = {
            "model_path": arguments.model_path,
            "dataset_name": arguments.dataset_name,
            "dataset_path": str(arguments.dataset_path),
            "seed": arguments.seed,
            "device": arguments.device,
            "requested_dtype": arguments.dtype,
            "controller_endpoint": (
                arguments.controller_endpoint if arguments.mode == "expertkit" else None
            ),
            "instance_id": arguments.instance_id if arguments.mode == "expertkit" else None,
        }
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
