"""Tests for vLLM-compatible ShareGPT prompt selection and batching."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from expertkit_torch.benchmark.datasets import BenchmarkSample, ShareGPTDataset


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, tokens: dict[str, list[int]]) -> None:
        self.tokens = tokens

    def __call__(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=self.tokens[text])


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "first",
                    "conversations": [
                        {"from": "human", "value": "short"},
                        {"from": "gpt", "value": "ignored completion"},
                    ],
                },
                {
                    "id": "second",
                    "conversations": [
                        {"from": "human", "value": "long"},
                        {"from": "gpt", "value": "second completion"},
                        {"from": "human", "value": "ignored third turn"},
                    ],
                },
                {
                    "id": "third",
                    "conversations": [
                        {"from": "human", "value": "medium"},
                        {"from": "gpt", "value": "third completion"},
                    ],
                },
                {
                    "id": "incomplete",
                    "conversations": [{"from": "human", "value": "one turn"}],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_sharegpt_selects_valid_prompts_and_left_pads_batches(tmp_path: Path) -> None:
    source = tmp_path / "sharegpt.json"
    _write_dataset(source)
    tokenizer = FakeTokenizer(
        {
            "short": [1, 2, 3],
            "long": [10, 11, 12, 13, 14, 15],
            "medium": [20, 21, 22, 23],
        }
    )
    dataset = ShareGPTDataset(source, seed=0)
    assert dataset.samples[1] == BenchmarkSample(
        prompt="long",
        completion="second completion",
    )

    batches = tuple(
        dataset.iter_batches(
            tokenizer,
            batch_size=2,
            num_prompts=2,
            output_length=8,
            device=torch.device("cpu"),
        )
    )

    assert len(batches) == 1
    assert batches[0].input_ids.tolist() == [
        [0, 0, 20, 21, 22, 23],
        [10, 11, 12, 13, 14, 15],
    ]
    assert batches[0].attention_mask.tolist() == [
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]
    assert batches[0].batch_size == 2
    assert batches[0].input_tokens == 10


def test_sharegpt_reports_when_too_few_prompts_pass_vllm_limits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sharegpt.json"
    _write_dataset(source)
    tokenizer = FakeTokenizer(
        {
            "short": [1, 2, 3],
            "long": [1, 2, 3, 4],
            "medium": [1, 2, 3, 4],
        }
    )

    with pytest.raises(ValueError, match="only 2 usable prompts; offset 0 plus 3 prompts"):
        tuple(
            ShareGPTDataset(source).iter_batches(
                tokenizer,
                batch_size=1,
                num_prompts=3,
                output_length=8,
                device=torch.device("cpu"),
            )
        )


def test_sharegpt_offsets_select_disjoint_deterministic_prompts(tmp_path: Path) -> None:
    source = tmp_path / "sharegpt.json"
    _write_dataset(source)
    tokenizer = FakeTokenizer(
        {
            "short": [1, 2, 3, 4],
            "long": [10, 11, 12, 13, 14, 15],
            "medium": [20, 21, 22, 23],
        }
    )

    first = ShareGPTDataset(source, seed=7, offset=0)._select_samples(
        tokenizer,
        num_prompts=1,
        output_length=8,
    )
    second = ShareGPTDataset(source, seed=7, offset=1)._select_samples(
        tokenizer,
        num_prompts=1,
        output_length=8,
    )

    assert first[0].sample != second[0].sample
