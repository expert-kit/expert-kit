"""Measure dataset-driven prefill and cached greedy decode."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from expertkit_torch.benchmark.datasets import BenchmarkDataset


@dataclass(frozen=True)
class RunMetrics:
    """Timing and throughput from one measured model-input batch."""

    batch_size: int
    input_tokens: int
    padded_input_length: int
    output_length: int
    prefill_seconds: float
    decode_seconds: float
    generated_token_ids: tuple[tuple[int, ...], ...] = ()
    generated_text: tuple[str, ...] = ()

    @property
    def total_seconds(self) -> float:
        """Return prefill plus decode wall time."""

        return self.prefill_seconds + self.decode_seconds

    @property
    def prefill_tps(self) -> float:
        """Return non-padding input tokens processed per second."""

        return self.input_tokens / self.prefill_seconds

    @property
    def decode_tps(self) -> float | None:
        """Return aggregate decode tokens per second, excluding the first token."""

        decode_tokens = self.batch_size * (self.output_length - 1)
        if decode_tokens == 0:
            return None
        return decode_tokens / self.decode_seconds

    @property
    def decode_step_ms(self) -> float | None:
        """Return mean wall time for one batched decode step."""

        decode_steps = self.output_length - 1
        if decode_steps == 0:
            return None
        return self.decode_seconds * 1000 / decode_steps

    @property
    def output_tps(self) -> float:
        """Return all generated tokens per complete-generation second."""

        return self.batch_size * self.output_length / self.total_seconds

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible raw and derived metrics."""

        return {
            **asdict(self),
            "total_seconds": self.total_seconds,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "decode_step_ms": self.decode_step_ms,
            "output_tps": self.output_tps,
        }


@dataclass(frozen=True)
class BatchBenchmark:
    """Measurements collected for one configured static batch size."""

    batch_size: int
    measurements: tuple[RunMetrics, ...]

    def median(self) -> dict[str, int | float | None]:
        """Return the median of every reported timing and throughput."""

        first = self.measurements[0]
        decode_tps = [
            value
            for measurement in self.measurements
            if (value := measurement.decode_tps) is not None
        ]
        decode_step_ms = [
            value
            for measurement in self.measurements
            if (value := measurement.decode_step_ms) is not None
        ]
        return {
            "batch_size": self.batch_size,
            "input_tokens": statistics.median(
                measurement.input_tokens for measurement in self.measurements
            ),
            "padded_input_length": statistics.median(
                measurement.padded_input_length for measurement in self.measurements
            ),
            "output_length": first.output_length,
            "prefill_seconds": statistics.median(
                measurement.prefill_seconds for measurement in self.measurements
            ),
            "decode_seconds": statistics.median(
                measurement.decode_seconds for measurement in self.measurements
            ),
            "total_seconds": statistics.median(
                measurement.total_seconds for measurement in self.measurements
            ),
            "prefill_tps": statistics.median(
                measurement.prefill_tps for measurement in self.measurements
            ),
            "decode_tps": statistics.median(decode_tps) if decode_tps else None,
            "decode_step_ms": (statistics.median(decode_step_ms) if decode_step_ms else None),
            "output_tps": statistics.median(
                measurement.output_tps for measurement in self.measurements
            ),
        }

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible measurements and their summary."""

        return {
            "batch_size": self.batch_size,
            "measurements": [measurement.as_dict() for measurement in self.measurements],
            "median": self.median(),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark result for one loaded model and dataset."""

    model_type: str
    mode: str
    num_prompts: int
    output_length: int
    warmup_runs: int
    batches: tuple[BatchBenchmark, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible benchmark report."""

        return {
            "schema_version": 2,
            "model_type": self.model_type,
            "mode": self.mode,
            "num_prompts": self.num_prompts,
            "output_length": self.output_length,
            "warmup_runs": self.warmup_runs,
            "batches": [batch.as_dict() for batch in self.batches],
        }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize()


def _measure_generation(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    output_length: int,
    clock: Callable[[], float],
    synchronize: Callable[[torch.device], None],
) -> RunMetrics:
    device = input_ids.device
    input_tokens = int(attention_mask.sum().item())

    synchronize(device)
    prefill_start = clock()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [next_token]
    synchronize(device)
    prefill_seconds = clock() - prefill_start
    if prefill_seconds <= 0:
        raise RuntimeError("prefill timer did not advance")

    decode_seconds = 0.0
    if output_length > 1:
        past_key_values = output.past_key_values
        if past_key_values is None:
            raise RuntimeError("model did not return a key/value cache")
        decode_start = clock()
        for _ in range(output_length - 1):
            attention_mask = torch.cat(
                (
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ),
                dim=1,
            )
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
        synchronize(device)
        decode_seconds = clock() - decode_start
        if decode_seconds <= 0:
            raise RuntimeError("decode timer did not advance")

    generated_token_ids = tuple(
        tuple(row) for row in torch.cat(generated_tokens, dim=1).detach().cpu().tolist()
    )
    generated_text = tuple(
        tokenizer.batch_decode(
            generated_token_ids,
            skip_special_tokens=False,
        )
    )
    return RunMetrics(
        batch_size=input_ids.shape[0],
        input_tokens=input_tokens,
        padded_input_length=input_ids.shape[1],
        output_length=output_length,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        generated_token_ids=generated_token_ids,
        generated_text=generated_text,
    )


def _validate_benchmark_arguments(
    *,
    batch_sizes: Sequence[int],
    num_prompts: int,
    output_length: int,
    warmup_runs: int,
) -> None:
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    if any(isinstance(value, bool) or value <= 0 for value in batch_sizes):
        raise ValueError("every batch size must be positive")
    if isinstance(num_prompts, bool) or num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if isinstance(output_length, bool) or output_length <= 0:
        raise ValueError("output_length must be positive")
    if isinstance(warmup_runs, bool) or warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")


def run_benchmark(
    model: Any,
    tokenizer: Any,
    dataset: BenchmarkDataset,
    *,
    model_type: str,
    mode: str,
    batch_sizes: Sequence[int],
    num_prompts: int,
    output_length: int,
    warmup_runs: int,
    device: str | torch.device,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[torch.device], None] = _synchronize,
    before_measurement: Callable[[], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> BenchmarkReport:
    """Measure each selected dataset prompt once for every static batch size."""

    _validate_benchmark_arguments(
        batch_sizes=batch_sizes,
        num_prompts=num_prompts,
        output_length=output_length,
        warmup_runs=warmup_runs,
    )
    resolved_device = torch.device(device)
    batch_results: list[BatchBenchmark] = []

    with torch.inference_mode():
        for batch_size in batch_sizes:
            model_inputs = tuple(
                dataset.iter_batches(
                    tokenizer,
                    batch_size=batch_size,
                    num_prompts=num_prompts,
                    output_length=output_length,
                    device=resolved_device,
                )
            )
            if not model_inputs:
                raise ValueError("dataset produced no model-input batches")

            warmup_input = model_inputs[0]
            for _ in range(warmup_runs):
                _measure_generation(
                    model,
                    tokenizer,
                    warmup_input.input_ids,
                    attention_mask=warmup_input.attention_mask,
                    output_length=output_length,
                    clock=clock,
                    synchronize=synchronize,
                )

            if before_measurement is not None:
                before_measurement()

            measurements: list[RunMetrics] = []
            for model_input in model_inputs:
                measurement = _measure_generation(
                    model,
                    tokenizer,
                    model_input.input_ids,
                    attention_mask=model_input.attention_mask,
                    output_length=output_length,
                    clock=clock,
                    synchronize=synchronize,
                )
                measurements.append(measurement)
                if on_progress is not None:
                    on_progress(measurement.batch_size)
            batch_results.append(BatchBenchmark(batch_size, tuple(measurements)))

    return BenchmarkReport(
        model_type=model_type,
        mode=mode,
        num_prompts=num_prompts,
        output_length=output_length,
        warmup_runs=warmup_runs,
        batches=tuple(batch_results),
    )
