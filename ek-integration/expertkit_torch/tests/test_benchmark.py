"""Tests for fixed-length prefill and decode benchmarking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from expertkit_torch.benchmark.runner import (
    BatchBenchmark,
    RunMetrics,
    build_fixed_input,
    run_benchmark,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, _: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [4, 5, 6]

    def batch_decode(
        self,
        token_ids: tuple[tuple[int, ...], ...],
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        assert not skip_special_tokens
        return [" ".join(str(token_id) for token_id in row) for row in token_ids]


class FakeModel:
    def __init__(self) -> None:
        self.input_lengths: list[int] = []

    def __call__(self, **arguments: Any) -> SimpleNamespace:
        input_ids = arguments["input_ids"]
        self.input_lengths.append(input_ids.shape[1])
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 8))
        return SimpleNamespace(logits=logits, past_key_values=object())


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current


def test_fixed_input_has_exact_shape_and_repeated_rows() -> None:
    result = build_fixed_input(
        FakeTokenizer(),
        batch_size=2,
        input_length=5,
        device=torch.device("cpu"),
    )

    assert result.tolist() == [[4, 5, 6, 4, 5], [4, 5, 6, 4, 5]]


def test_benchmark_runs_prefill_and_cached_decode_for_each_batch() -> None:
    model = FakeModel()
    synchronized: list[torch.device] = []

    report = run_benchmark(
        model,
        FakeTokenizer(),
        model_type="qwen3_moe",
        mode="expertkit",
        batch_sizes=(1, 3),
        input_length=5,
        output_length=4,
        warmup_runs=0,
        measured_runs=1,
        device="cpu",
        clock=IncrementingClock(),
        synchronize=synchronized.append,
    )

    assert model.input_lengths == [5, 1, 1, 1, 5, 1, 1, 1]
    assert synchronized == [torch.device("cpu")] * 6
    assert [batch.batch_size for batch in report.batches] == [1, 3]
    first = report.batches[0].runs[0]
    assert first.prefill_seconds == 1
    assert first.decode_seconds == 1
    assert first.prefill_tps == 5
    assert first.decode_tps == 3
    assert first.decode_step_ms == pytest.approx(1000 / 3)
    assert first.output_tps == 2
    assert first.generated_token_ids == ((0, 0, 0, 0),)
    assert first.generated_text == ("0 0 0 0",)


def test_output_length_one_has_no_decode_phase() -> None:
    model = FakeModel()

    report = run_benchmark(
        model,
        FakeTokenizer(),
        model_type="mixtral",
        mode="local",
        batch_sizes=(2,),
        input_length=3,
        output_length=1,
        warmup_runs=0,
        measured_runs=1,
        device="cpu",
        clock=IncrementingClock(),
        synchronize=lambda _: None,
    )

    run = report.batches[0].runs[0]
    assert model.input_lengths == [3]
    assert run.decode_seconds == 0
    assert run.decode_tps is None
    assert run.decode_step_ms is None
    assert report.as_dict()["batches"][0]["median"]["decode_tps"] is None


def test_warmups_are_excluded_from_reported_runs() -> None:
    model = FakeModel()

    report = run_benchmark(
        model,
        FakeTokenizer(),
        model_type="qwen3_moe",
        mode="local",
        batch_sizes=(1,),
        input_length=2,
        output_length=1,
        warmup_runs=2,
        measured_runs=3,
        device="cpu",
        clock=IncrementingClock(),
        synchronize=lambda _: None,
    )

    assert model.input_lengths == [2, 2, 2, 2, 2]
    assert len(report.batches[0].runs) == 3


def test_batch_summary_uses_medians_and_keeps_raw_runs() -> None:
    batch = BatchBenchmark(
        2,
        (
            RunMetrics(2, 4, 3, 1.0, 3.0),
            RunMetrics(2, 4, 3, 3.0, 1.0),
            RunMetrics(2, 4, 3, 2.0, 2.0),
        ),
    )

    summary = batch.median()

    assert summary["prefill_seconds"] == 2
    assert summary["decode_seconds"] == 2
    assert summary["total_seconds"] == 4
    assert len(batch.as_dict()["runs"]) == 3


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"batch_sizes": ()}, "batch_sizes"),
        ({"batch_sizes": (0,)}, "batch size"),
        ({"input_length": 0}, "input_length"),
        ({"output_length": 0}, "output_length"),
        ({"warmup_runs": -1}, "warmup_runs"),
        ({"measured_runs": 0}, "measured_runs"),
    ],
)
def test_benchmark_rejects_invalid_limits(
    overrides: dict[str, Any],
    message: str,
) -> None:
    arguments = {
        "model_type": "qwen3_moe",
        "mode": "expertkit",
        "batch_sizes": (1,),
        "input_length": 2,
        "output_length": 2,
        "warmup_runs": 0,
        "measured_runs": 1,
        "device": "cpu",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        run_benchmark(FakeModel(), FakeTokenizer(), **arguments)
