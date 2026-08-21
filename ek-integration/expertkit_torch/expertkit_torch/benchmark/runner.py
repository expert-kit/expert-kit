"""Measure fixed-length prefill and cached greedy decode."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from expertkit_transport import frontend_request_span

_FIXED_INPUT_TEXT = "Expert Kit routed mixture of experts benchmark input"


@dataclass(frozen=True)
class RunMetrics:
    """Timing and throughput from one measured generation."""

    batch_size: int
    input_length: int
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
        """Return aggregate input tokens processed per second."""

        return self.batch_size * self.input_length / self.prefill_seconds

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
    """Raw runs and median metrics for one batch size."""

    batch_size: int
    runs: tuple[RunMetrics, ...]

    def median(self) -> dict[str, int | float | None]:
        """Return the median of every reported timing and throughput."""

        first = self.runs[0]
        decode_tps = [value for run in self.runs if (value := run.decode_tps) is not None]
        decode_step_ms = [value for run in self.runs if (value := run.decode_step_ms) is not None]
        return {
            "batch_size": self.batch_size,
            "input_length": first.input_length,
            "output_length": first.output_length,
            "prefill_seconds": statistics.median(run.prefill_seconds for run in self.runs),
            "decode_seconds": statistics.median(run.decode_seconds for run in self.runs),
            "total_seconds": statistics.median(run.total_seconds for run in self.runs),
            "prefill_tps": statistics.median(run.prefill_tps for run in self.runs),
            "decode_tps": statistics.median(decode_tps) if decode_tps else None,
            "decode_step_ms": statistics.median(decode_step_ms) if decode_step_ms else None,
            "output_tps": statistics.median(run.output_tps for run in self.runs),
        }

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible raw runs and their summary."""

        return {
            "batch_size": self.batch_size,
            "runs": [run.as_dict() for run in self.runs],
            "median": self.median(),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark result for one loaded model."""

    model_type: str
    mode: str
    input_length: int
    output_length: int
    warmup_runs: int
    measured_runs: int
    batches: tuple[BatchBenchmark, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible benchmark report."""

        return {
            "schema_version": 1,
            "model_type": self.model_type,
            "mode": self.mode,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "batches": [batch.as_dict() for batch in self.batches],
        }


def build_fixed_input(
    tokenizer: Any,
    *,
    batch_size: int,
    input_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build equal, deterministic token rows with exactly the requested length."""

    token_ids = tokenizer.encode(_FIXED_INPUT_TEXT, add_special_tokens=False)
    if not token_ids:
        fallback_id = tokenizer.bos_token_id
        if fallback_id is None:
            fallback_id = tokenizer.eos_token_id
        if fallback_id is None:
            raise ValueError("tokenizer produced no tokens and has no BOS or EOS token")
        token_ids = [fallback_id]
    repeated = (token_ids * ((input_length + len(token_ids) - 1) // len(token_ids)))[:input_length]
    row = torch.tensor(repeated, dtype=torch.long, device=device)
    return row.unsqueeze(0).repeat(batch_size, 1)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _measure_generation(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    output_length: int,
    clock: Callable[[], float],
    synchronize: Callable[[torch.device], None],
) -> RunMetrics:
    attributes = {
        "expertkit.batch_size": int(input_ids.shape[0]),
        "expertkit.input_length": int(input_ids.shape[1]),
        "expertkit.output_length": output_length,
    }
    with frontend_request_span(attributes=attributes):
        return _measure_generation_body(
            model,
            tokenizer,
            input_ids,
            output_length=output_length,
            clock=clock,
            synchronize=synchronize,
        )


def _measure_generation_body(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    output_length: int,
    clock: Callable[[], float],
    synchronize: Callable[[torch.device], None],
) -> RunMetrics:
    device = input_ids.device
    attention_mask = torch.ones_like(input_ids)

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
                    torch.ones(
                        (input_ids.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
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
        input_length=input_ids.shape[1],
        output_length=output_length,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        generated_token_ids=generated_token_ids,
        generated_text=generated_text,
    )


def _validate_benchmark_arguments(
    *,
    batch_sizes: Sequence[int],
    input_length: int,
    output_length: int,
    warmup_runs: int,
    measured_runs: int,
) -> None:
    if not batch_sizes:
        raise ValueError("batch_sizes must not be empty")
    if any(isinstance(value, bool) or value <= 0 for value in batch_sizes):
        raise ValueError("every batch size must be positive")
    if isinstance(input_length, bool) or input_length <= 0:
        raise ValueError("input_length must be positive")
    if isinstance(output_length, bool) or output_length <= 0:
        raise ValueError("output_length must be positive")
    if isinstance(warmup_runs, bool) or warmup_runs < 0:
        raise ValueError("warmup_runs must not be negative")
    if isinstance(measured_runs, bool) or measured_runs <= 0:
        raise ValueError("measured_runs must be positive")


def run_benchmark(
    model: Any,
    tokenizer: Any,
    *,
    model_type: str,
    mode: str,
    batch_sizes: Sequence[int],
    input_length: int,
    output_length: int,
    warmup_runs: int,
    measured_runs: int,
    device: str | torch.device,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[torch.device], None] = _synchronize,
) -> BenchmarkReport:
    """Run fixed-length generations and summarize phase-level performance."""

    _validate_benchmark_arguments(
        batch_sizes=batch_sizes,
        input_length=input_length,
        output_length=output_length,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    resolved_device = torch.device(device)
    batches: list[BatchBenchmark] = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            input_ids = build_fixed_input(
                tokenizer,
                batch_size=batch_size,
                input_length=input_length,
                device=resolved_device,
            )
            for _ in range(warmup_runs):
                _measure_generation(
                    model,
                    tokenizer,
                    input_ids,
                    output_length=output_length,
                    clock=clock,
                    synchronize=synchronize,
                )
            runs = tuple(
                _measure_generation(
                    model,
                    tokenizer,
                    input_ids,
                    output_length=output_length,
                    clock=clock,
                    synchronize=synchronize,
                )
                for _ in range(measured_runs)
            )
            batches.append(BatchBenchmark(batch_size, runs))
    return BenchmarkReport(
        model_type=model_type,
        mode=mode,
        input_length=input_length,
        output_length=output_length,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        batches=tuple(batches),
    )
