"""Tests for dataset-driven prefill and decode benchmarking."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from expertkit_torch.benchmark.datasets import ModelInputBatch
from expertkit_torch.benchmark.runner import (
    BatchBenchmark,
    RunMetrics,
    run_benchmark,
)


class FakeTokenizer:
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
        logits = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], 8),
            device=input_ids.device,
        )
        return SimpleNamespace(logits=logits, past_key_values=object())


class FakeDataset:
    def __init__(self, *model_inputs: ModelInputBatch) -> None:
        self.model_inputs = model_inputs
        self.calls: list[dict[str, object]] = []

    def iter_batches(
        self,
        tokenizer: Any,
        *,
        batch_size: int,
        num_prompts: int,
        output_length: int,
        device: torch.device,
    ) -> Iterator[ModelInputBatch]:
        self.calls.append(
            {
                "tokenizer": tokenizer,
                "batch_size": batch_size,
                "num_prompts": num_prompts,
                "output_length": output_length,
                "device": device,
            }
        )
        yield from self.model_inputs


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        current = self.value
        self.value += 1.0
        return current


def _model_input(
    input_ids: list[list[int]],
    attention_mask: list[list[int]],
) -> ModelInputBatch:
    return ModelInputBatch(
        input_ids=torch.tensor(input_ids),
        attention_mask=torch.tensor(attention_mask),
    )


def test_benchmark_warms_up_first_batch_then_measures_each_dataset_batch() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    dataset = FakeDataset(
        _model_input([[4, 5]], [[1, 1]]),
        _model_input([[6, 7, 8]], [[1, 1, 1]]),
    )
    synchronized: list[torch.device] = []

    report = run_benchmark(
        model,
        tokenizer,
        dataset,
        model_type="qwen3_moe",
        mode="expertkit",
        batch_sizes=(1,),
        num_prompts=2,
        output_length=4,
        warmup_runs=1,
        device="cpu",
        clock=IncrementingClock(),
        synchronize=synchronized.append,
    )

    assert model.input_lengths == [2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1]
    assert synchronized == [torch.device("cpu")] * 9
    assert dataset.calls == [
        {
            "tokenizer": tokenizer,
            "batch_size": 1,
            "num_prompts": 2,
            "output_length": 4,
            "device": torch.device("cpu"),
        }
    ]
    measurements = report.batches[0].measurements
    assert len(measurements) == 2
    assert [measurement.input_tokens for measurement in measurements] == [2, 3]
    assert measurements[0].prefill_tps == 2
    assert measurements[0].decode_tps == 3
    assert measurements[0].decode_step_ms == pytest.approx(1000 / 3)
    assert measurements[0].output_tps == 2
    assert measurements[0].generated_token_ids == ((0, 0, 0, 0),)
    assert measurements[0].generated_text == ("0 0 0 0",)


def test_output_length_one_has_no_decode_phase() -> None:
    model = FakeModel()
    dataset = FakeDataset(_model_input([[4, 5, 6]], [[1, 1, 1]]))

    report = run_benchmark(
        model,
        FakeTokenizer(),
        dataset,
        model_type="mixtral",
        mode="local",
        batch_sizes=(1,),
        num_prompts=1,
        output_length=1,
        warmup_runs=0,
        device="cpu",
        clock=IncrementingClock(),
        synchronize=lambda _: None,
    )

    measurement = report.batches[0].measurements[0]
    assert model.input_lengths == [3]
    assert measurement.decode_seconds == 0
    assert measurement.decode_tps is None
    assert measurement.decode_step_ms is None
    assert report.as_dict()["batches"][0]["median"]["decode_tps"] is None


def test_warmups_are_excluded_from_measurements() -> None:
    model = FakeModel()
    dataset = FakeDataset(_model_input([[4, 5]], [[1, 1]]))

    report = run_benchmark(
        model,
        FakeTokenizer(),
        dataset,
        model_type="qwen3_moe",
        mode="local",
        batch_sizes=(1,),
        num_prompts=1,
        output_length=1,
        warmup_runs=2,
        device="cpu",
        clock=IncrementingClock(),
        synchronize=lambda _: None,
    )

    assert model.input_lengths == [2, 2, 2]
    assert len(report.batches[0].measurements) == 1


def test_progress_reports_measured_batch_sizes_but_not_warmups() -> None:
    progress: list[int] = []
    report = run_benchmark(
        FakeModel(),
        FakeTokenizer(),
        FakeDataset(
            _model_input([[4, 5], [6, 7]], [[1, 1], [1, 1]]),
            _model_input([[8, 9, 10]], [[1, 1, 1]]),
        ),
        model_type="qwen3_moe",
        mode="local",
        batch_sizes=(2,),
        num_prompts=3,
        output_length=1,
        warmup_runs=1,
        device="cpu",
        synchronize=lambda _: None,
        on_progress=progress.append,
    )

    assert progress == [2, 1]
    assert sum(progress) == report.num_prompts


def test_batch_summary_uses_medians_and_keeps_raw_measurements() -> None:
    batch = BatchBenchmark(
        2,
        (
            RunMetrics(2, 8, 4, 3, 1.0, 3.0),
            RunMetrics(2, 10, 5, 3, 3.0, 1.0),
            RunMetrics(2, 12, 6, 3, 2.0, 2.0),
        ),
    )

    summary = batch.median()

    assert summary["input_tokens"] == 10
    assert summary["padded_input_length"] == 5
    assert summary["prefill_seconds"] == 2
    assert summary["decode_seconds"] == 2
    assert summary["total_seconds"] == 4
    assert len(batch.as_dict()["measurements"]) == 3


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"batch_sizes": ()}, "batch_sizes"),
        ({"batch_sizes": (0,)}, "batch size"),
        ({"num_prompts": 0}, "num_prompts"),
        ({"output_length": 0}, "output_length"),
        ({"warmup_runs": -1}, "warmup_runs"),
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
        "num_prompts": 1,
        "output_length": 2,
        "warmup_runs": 0,
        "device": "cpu",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        run_benchmark(
            FakeModel(),
            FakeTokenizer(),
            FakeDataset(_model_input([[4, 5]], [[1, 1]])),
            **arguments,
        )
