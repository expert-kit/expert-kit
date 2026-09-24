#!/usr/bin/env python3
"""Microbenchmark one EK computation backend with deterministic synthetic data."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import save

from expertkit_worker.backends import BackendBatch, ComputeBackend
from expertkit_worker.backends.factory import (
    create_compute_backend,
    create_weight_adapter,
    torch_dtype,
)
from expertkit_worker.config import WorkerConfig
from expertkit_worker.device import CpuWorkerRuntime, CudaWorkerRuntime, WorkerDeviceRuntime
from expertkit_worker.weights import ReadyWeightTable, WeightAdapter, parse_safetensors


@dataclass(slots=True)
class _Setup:
    config: WorkerConfig
    runtime: WorkerDeviceRuntime
    adapter: WeightAdapter[Any, Any]
    table: ReadyWeightTable[Any]
    backend: ComputeBackend
    batches: list[BackendBatch]
    outputs: list[torch.Tensor]
    source_values: list[dict[str, torch.Tensor]]
    source_buffers: list[bytearray]
    cpu_weights: list[Any]

    def close(self) -> None:
        for expert_id in range(self.config.model.experts_per_layer):
            self.table.begin_withdrawal(0, expert_id)
            ready = self.table.finish_withdrawal(0, expert_id)
            self.adapter.release_ready_weight(ready)


class _SubmissionGroup:
    def __init__(self, setup: _Setup) -> None:
        self._setup = setup
        concurrency = len(setup.batches)
        self._pool = (
            ThreadPoolExecutor(max_workers=concurrency)
            if concurrency > 1 and setup.runtime.device.type == "cpu"
            else None
        )
        if concurrency > 1 and self._pool is None:
            create_stream = getattr(setup.runtime, "create_stream", None)
            if create_stream is None:
                raise ValueError("concurrency > 1 requires an asynchronous device runtime")
            self._streams = [create_stream() for _ in range(concurrency)]
        else:
            self._streams = []

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()

    def run(self) -> None:
        if self._pool is not None:
            futures = [
                self._pool.submit(self._submit_and_wait, batch, output)
                for batch, output in zip(
                    self._setup.batches,
                    self._setup.outputs,
                    strict=True,
                )
            ]
            for future in futures:
                future.result()
            return

        if not self._streams:
            self._submit_and_wait(self._setup.batches[0], self._setup.outputs[0])
            return

        completions = []
        try:
            for stream, batch, output in zip(
                self._streams,
                self._setup.batches,
                self._setup.outputs,
                strict=True,
            ):
                stream_context = self._setup.runtime.stream_context  # type: ignore[attr-defined]
                with stream_context(stream):
                    completions.append(self._setup.backend.submit(batch, output))
            for completion in completions:
                completion.wait_host()
        finally:
            for completion in completions:
                completion.close()

    def _submit_and_wait(self, batch: BackendBatch, output: torch.Tensor) -> None:
        completion = self._setup.backend.submit(batch, output)
        try:
            completion.wait_host()
        finally:
            completion.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("torch", "ggml", "fused", "ascend_fused"),
        default="ascend_fused",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--quantization",
        choices=("none", "modelslim-w8a8-dynamic"),
        default="none",
    )
    parser.add_argument("--expert-compute", choices=("swiglu", "deepseek_v4"), default="swiglu")
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--intermediate-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--distribution", choices=("balanced", "hot"), default="balanced")
    parser.add_argument("--inactive-rate", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--max-relative-l2", type=float, default=0.06)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=Path, default=Path("benchmark-traces"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "tokens",
        "hidden_dim",
        "intermediate_dim",
        "top_k",
        "experts",
        "concurrency",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.top_k > args.experts:
        parser.error("--top-k cannot exceed --experts")
    if not 0 <= args.inactive_rate < 1:
        parser.error("--inactive-rate must be in [0, 1)")
    if args.max_relative_l2 < 0:
        parser.error("--max-relative-l2 must be nonnegative")
    if not -1 <= args.min_cosine <= 1:
        parser.error("--min-cosine must be in [-1, 1]")
    if args.expert_compute == "swiglu" and args.swiglu_limit != 0:
        parser.error("--swiglu-limit requires --expert-compute deepseek_v4")
    return args


def _make_config(args: argparse.Namespace) -> WorkerConfig:
    model: dict[str, Any] = {
        "instance_id": 1,
        "name": "benchmark/model",
        "weight_version": f"synthetic-seed-{args.seed}",
        "num_layers": 1,
        "experts_per_layer": args.experts,
        "hidden_dim": args.hidden_dim,
        "expert_intermediate_dim": args.intermediate_dim,
        "top_k": args.top_k,
        "activation_dtype": args.dtype,
        "weight_dtype": args.dtype,
        "expert_compute": args.expert_compute,
        "swiglu_limit": args.swiglu_limit,
    }
    if args.quantization != "none":
        model["quantization"] = {
            "type": args.quantization,
            "bits": 8,
            "group_size": None,
        }
    worker: dict[str, Any] = {
        "id": "backend-benchmark",
        "backend": args.backend,
        "device": args.device,
        "max_batch_tokens": args.tokens,
        "max_active_batches_per_device": args.concurrency,
        "device_memory_limit": "64GiB",
    }
    if args.backend == "ggml":
        worker["ggml"] = {"cpu_threads": max(1, torch.get_num_threads())}
    return WorkerConfig.model_validate(
        {
            "model": model,
            "worker": worker,
            "transport": {
                "type": "grpc",
                "max_pending_batches_per_device": args.concurrency,
                "listen": "127.0.0.1:50051",
                "advertise": "benchmark:50051",
            },
            "controller": {"endpoint": "127.0.0.1:50050"},
            "weight_manager": {
                "max_concurrent_loads": 1,
                "disk_cache": {"path": "/tmp/ek-backend-benchmark-cache"},
                "peer": {
                    "listen": "127.0.0.1:50052",
                    "advertise": "http://benchmark:50052",
                },
                "weight_server_endpoint": "http://127.0.0.1:50053",
            },
        }
    )


def _make_runtime(device_name: str) -> WorkerDeviceRuntime:
    if device_name == "cpu":
        return CpuWorkerRuntime(torch.device("cpu"))
    if device_name.startswith("cuda:"):
        runtime = CudaWorkerRuntime(torch.device(device_name))
        runtime.set_current_device()
        return runtime
    if device_name.startswith("npu:"):
        from expertkit_worker.device.ascend import AscendWorkerRuntime

        runtime = AscendWorkerRuntime(torch.device(device_name))
        runtime.set_current_device()
        return runtime
    raise ValueError(f"unsupported benchmark device: {device_name}")


def _make_source_values(
    *,
    expert_id: int,
    hidden_dim: int,
    intermediate_dim: int,
    dtype: torch.dtype,
    quantized: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + expert_id * 17)
    prefix = f"layers.0.ffn.experts.{expert_id}"
    shapes = (
        ("w1", (intermediate_dim, hidden_dim)),
        ("w3", (intermediate_dim, hidden_dim)),
        ("w2", (hidden_dim, intermediate_dim)),
    )
    values: dict[str, torch.Tensor] = {}
    for name, shape in shapes:
        if quantized:
            values[f"{prefix}.{name}.weight"] = torch.randint(
                -16,
                17,
                shape,
                generator=generator,
                dtype=torch.int8,
            )
            scale = torch.rand((shape[0], 1), generator=generator, dtype=torch.float32)
            values[f"{prefix}.{name}.weight_scale"] = scale.mul_(0.015).add_(0.005)
            values[f"{prefix}.{name}.weight_offset"] = torch.zeros(
                (shape[0], 1),
                dtype=torch.float32,
            )
        else:
            scale = 0.03 / math.sqrt(shape[1])
            values[f"{prefix}.{name}.weight"] = (
                torch.randn(shape, generator=generator, dtype=torch.float32) * scale
            ).to(dtype)
    return values


def _make_batch(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: torch.device,
    slot: int,
) -> BackendBatch:
    generator = torch.Generator().manual_seed(args.seed + 10_000 + slot)
    hidden = torch.randn(
        (args.tokens, args.hidden_dim),
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.1)
    ids = torch.empty((args.tokens, args.top_k), dtype=torch.int32)
    for token in range(args.tokens):
        base = token * args.top_k if args.distribution == "balanced" else (token // 5) * args.top_k
        for route in range(args.top_k):
            ids[token, route] = (base + route) % args.experts
    if args.inactive_rate:
        mask = torch.rand(ids.shape, generator=generator) < args.inactive_rate
        mask[:, 0] = False
        ids.masked_fill_(mask, -1)
    routing = torch.rand(ids.shape, generator=generator, dtype=torch.float32)
    routing.masked_fill_(ids.eq(-1), 0)
    routing.div_(routing.sum(dim=1, keepdim=True))
    distinct = tuple(sorted(set(ids[ids.ge(0)].tolist())))
    return BackendBatch(
        layer_id=0,
        hidden_states=hidden.to(device=device, dtype=dtype),
        expert_ids=ids.to(device=device),
        routing_weights=routing.to(device=device),
        distinct_expert_ids=distinct,
    )


def _build_setup(args: argparse.Namespace) -> _Setup:
    config = _make_config(args)
    runtime = _make_runtime(args.device)
    dtype = torch_dtype(config.model.activation_dtype)
    adapter = create_weight_adapter(
        config,
        source_dtype=torch_dtype(config.model.weight_dtype),
        compute_dtype=dtype,
        runtime=runtime,
    )
    adapter.initialize_ready_storage(args.experts)
    table: ReadyWeightTable[Any] = ReadyWeightTable(1, args.experts)
    source_values: list[dict[str, torch.Tensor]] = []
    source_buffers: list[bytearray] = []
    cpu_weights: list[Any] = []
    quantized = args.quantization != "none"
    with runtime.device_context():
        for expert_id in range(args.experts):
            values = _make_source_values(
                expert_id=expert_id,
                hidden_dim=args.hidden_dim,
                intermediate_dim=args.intermediate_dim,
                dtype=dtype,
                quantized=quantized,
                seed=args.seed,
            )
            buffer = bytearray(save(values))
            cpu_weight = adapter.make_cpu_weight(parse_safetensors(buffer))
            ready = adapter.make_ready_weight(cpu_weight, layer_id=0, expert_id=expert_id)
            table.publish(0, expert_id, ready)
            source_values.append(values)
            source_buffers.append(buffer)
            cpu_weights.append(cpu_weight)
        backend = create_compute_backend(
            config,
            dtype=dtype,
            runtime=runtime,
            acquire_many=table.acquire_many,
        )
        batches = [
            _make_batch(args, dtype=dtype, device=runtime.device, slot=slot)
            for slot in range(args.concurrency)
        ]
        outputs = [torch.empty_like(batch.hidden_states) for batch in batches]
    return _Setup(
        config=config,
        runtime=runtime,
        adapter=adapter,
        table=table,
        backend=backend,
        batches=batches,
        outputs=outputs,
        source_values=source_values,
        source_buffers=source_buffers,
        cpu_weights=cpu_weights,
    )


def _projection(
    source: dict[str, torch.Tensor],
    expert_id: int,
    name: str,
    *,
    quantized: bool,
) -> torch.Tensor:
    prefix = f"layers.0.ffn.experts.{expert_id}.{name}"
    weight = source[f"{prefix}.weight"].float()
    if quantized:
        weight = weight * source[f"{prefix}.weight_scale"].float()
    return weight


def _reference_output(setup: _Setup, batch: BackendBatch) -> torch.Tensor:
    hidden = batch.hidden_states.float().cpu()
    expert_ids = batch.expert_ids.cpu()
    routing = batch.routing_weights.cpu()
    output = torch.zeros_like(hidden)
    quantized = setup.config.model.quantization is not None
    for expert_id in batch.distinct_expert_ids:
        coordinates = torch.nonzero(expert_ids == expert_id, as_tuple=False)
        token_indices = coordinates[:, 0]
        route_indices = coordinates[:, 1]
        x = hidden.index_select(0, token_indices)
        source = setup.source_values[expert_id]
        gate = functional.linear(
            x,
            _projection(source, expert_id, "w1", quantized=quantized),
        )
        up = functional.linear(
            x,
            _projection(source, expert_id, "w3", quantized=quantized),
        )
        if setup.config.model.expert_compute == "deepseek_v4":
            limit = setup.config.model.swiglu_limit
            if limit > 0:
                gate.clamp_(max=limit)
                up.clamp_(min=-limit, max=limit)
        activated = functional.silu(gate) * up
        expert_output = functional.linear(
            activated,
            _projection(source, expert_id, "w2", quantized=quantized),
        )
        expert_output.mul_(routing[token_indices, route_indices].unsqueeze(1))
        output.index_add_(0, token_indices, expert_output)
    return output


def _check_correctness(
    args: argparse.Namespace,
    setup: _Setup,
    runner: _SubmissionGroup,
) -> dict[str, Any]:
    runner.run()
    quantized = setup.config.model.quantization is not None
    default_tolerance = 0.08 if quantized else (0.02 if args.dtype != "fp32" else 1e-5)
    atol = default_tolerance if args.atol is None else args.atol
    rtol = default_tolerance if args.rtol is None else args.rtol
    maximum = 0.0
    mean = 0.0
    maximum_relative_l2 = 0.0
    minimum_cosine = 1.0
    for batch_index, (batch, actual) in enumerate(
        zip(setup.batches, setup.outputs, strict=True)
    ):
        expected = _reference_output(setup, batch)
        actual_cpu = actual.float().cpu()
        difference = (actual_cpu - expected).abs()
        max_abs_error = difference.max().item()
        relative_l2 = float(
            difference.norm() / expected.norm().clamp_min(1e-10)
        )
        if torch.count_nonzero(expected) == 0:
            cosine = 1.0 if torch.count_nonzero(actual_cpu) == 0 else 0.0
        else:
            cosine = float(
                functional.cosine_similarity(
                    actual_cpu.flatten(),
                    expected.flatten(),
                    dim=0,
                )
            )
        maximum = max(maximum, max_abs_error)
        mean += difference.mean().item()
        maximum_relative_l2 = max(maximum_relative_l2, relative_l2)
        minimum_cosine = min(minimum_cosine, cosine)
        if quantized:
            valid = (
                torch.isfinite(actual_cpu).all()
                and relative_l2 < args.max_relative_l2
                and cosine > args.min_cosine
            )
        else:
            valid = torch.allclose(actual_cpu, expected, atol=atol, rtol=rtol)
        if not valid:
            raise RuntimeError(
                "backend correctness check failed: "
                f"batch={batch_index}, max_abs_error={max_abs_error:.6g}, "
                f"relative_l2={relative_l2:.6g}, cosine={cosine:.6g}, "
                f"atol={atol}, rtol={rtol}, "
                f"max_relative_l2={args.max_relative_l2}, min_cosine={args.min_cosine}"
            )
    return {
        "checked": True,
        "reference": "independent_fp32_dequantized" if quantized else "independent_fp32",
        "criterion": "relative_l2_and_cosine" if quantized else "allclose",
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": maximum,
        "mean_abs_error": mean / len(setup.batches),
        "max_relative_l2": maximum_relative_l2,
        "max_relative_l2_limit": args.max_relative_l2,
        "min_cosine_similarity": minimum_cosine,
        "min_cosine_similarity_limit": args.min_cosine,
    }


def _profile_context(args: argparse.Namespace, device: torch.device):
    if not args.profile:
        return nullcontext(None)
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "npu":
        from torch_npu import profiler

        return profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(str(args.profile_dir)),
            record_shapes=True,
        )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(args.profile_dir)),
        record_shapes=True,
    )


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "npu":
        import torch_npu

        return torch_npu.npu.max_memory_allocated(device)
    return None


def _memory_info(runtime: WorkerDeviceRuntime) -> tuple[int, int] | None:
    try:
        return runtime.memory_info()
    except RuntimeError:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    setup = _build_setup(args)
    runner = _SubmissionGroup(setup)
    try:
        correctness = (
            {"checked": False}
            if args.skip_correctness
            else _check_correctness(args, setup, runner)
        )
        for _ in range(args.warmup):
            runner.run()

        memory_before = _memory_info(setup.runtime)
        _reset_peak_memory(setup.runtime.device)
        durations_ms: list[float] = []
        with _profile_context(args, setup.runtime.device) as active_profiler:
            for _ in range(args.repeats):
                started = time.perf_counter_ns()
                runner.run()
                durations_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                if active_profiler is not None:
                    active_profiler.step()
        memory_after = _memory_info(setup.runtime)
        peak_allocated = _peak_memory(setup.runtime.device)
        mean_ms = statistics.fmean(durations_ms)
        total_tokens = args.tokens * args.concurrency
        estimate = setup.backend.estimate_resources(args.tokens)
        return {
            "schema_version": 1,
            "backend": args.backend,
            "device": args.device,
            "dtype": args.dtype,
            "quantization": args.quantization,
            "expert_compute": args.expert_compute,
            "swiglu_limit": args.swiglu_limit,
            "shape": {
                "tokens": args.tokens,
                "hidden_dim": args.hidden_dim,
                "intermediate_dim": args.intermediate_dim,
                "top_k": args.top_k,
                "experts": args.experts,
            },
            "routing": {
                "distribution": args.distribution,
                "inactive_rate": args.inactive_rate,
                "seed": args.seed,
            },
            "execution": {
                "concurrency": args.concurrency,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "profile": args.profile,
            },
            "correctness": correctness,
            "latency_ms": {
                "mean": mean_ms,
                "min": min(durations_ms),
                "p50": _percentile(durations_ms, 0.50),
                "p95": _percentile(durations_ms, 0.95),
                "max": max(durations_ms),
            },
            "throughput": {
                "tokens_per_second": total_tokens * 1000 / mean_ms,
                "batches_per_second": args.concurrency * 1000 / mean_ms,
            },
            "memory": {
                "free_bytes_before": None if memory_before is None else memory_before[0],
                "free_bytes_after": None if memory_after is None else memory_after[0],
                "total_bytes": None if memory_after is None else memory_after[1],
                "peak_allocated_bytes": peak_allocated,
                "estimated_temporary_bytes_per_active_batch": (
                    estimate.temporary_bytes_per_active_batch
                ),
                "estimated_total_temporary_bytes": estimate.total_bytes(args.concurrency),
            },
        }
    finally:
        runner.close()
        setup.close()


def main() -> None:
    args = _parse_args()
    result = _run_benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
