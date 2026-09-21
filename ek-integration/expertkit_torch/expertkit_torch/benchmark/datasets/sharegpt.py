from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, TypeAdapter

from .dataset import BenchmarkSample, ModelInputBatch


class ShareGPTTurn(BaseModel):
    role: str | None = Field(default=None, alias="from")
    value: str


class ShareGPTEntry(BaseModel):
    conversations: list[ShareGPTTurn]

    @property
    def prompt(self) -> str:
        return self.conversations[0].value

    @property
    def completion(self) -> str:
        return self.conversations[1].value


_SHARE_GPT_ENTRIES = TypeAdapter(list[ShareGPTEntry])


@dataclass(frozen=True, slots=True)
class _TokenizedSample:
    sample: BenchmarkSample
    input_ids: tuple[int, ...]


class ShareGPTDataset:
    def __init__(self, dataset: Path, *, seed: int = 0, offset: int = 0) -> None:
        if isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must not be negative")
        self.samples = _convert_to_benchmark_samples(dataset)
        self.seed = seed
        self.offset = offset

    def iter_batches(
        self,
        tokenizer: Any,
        *,
        batch_size: int,
        num_prompts: int,
        output_length: int,
        device: torch.device,
    ) -> Iterator[ModelInputBatch]:
        selected = self._select_samples(
            tokenizer,
            num_prompts=num_prompts,
            output_length=output_length,
        )
        for chunk in batched(selected, n=batch_size):
            yield _collate_left_padded(
                chunk,
                tokenizer=tokenizer,
                device=device,
            )

    def _select_samples(
        self,
        tokenizer: Any,
        *,
        num_prompts: int,
        output_length: int,
    ) -> list[_TokenizedSample]:
        candidates = list(self.samples)
        random.Random(self.seed).shuffle(candidates)

        selected: list[_TokenizedSample] = []
        valid_index = 0
        for sample in candidates:
            input_ids = tuple(tokenizer(sample.prompt).input_ids)
            prompt_length = len(input_ids)
            if not _is_valid_sequence(prompt_length, output_length):
                continue
            if valid_index < self.offset:
                valid_index += 1
                continue
            selected.append(_TokenizedSample(sample, input_ids))
            if len(selected) == num_prompts:
                return selected
            valid_index += 1

        raise ValueError(
            "ShareGPT contains only "
            f"{self.offset + len(selected)} usable prompts; "
            f"offset {self.offset} plus {num_prompts} prompts were requested"
        )


def _is_valid_sequence(prompt_length: int, output_length: int) -> bool:
    return 4 <= prompt_length <= 1024 and prompt_length + output_length <= 2048


def _collate_left_padded(
    samples: tuple[_TokenizedSample, ...],
    *,
    tokenizer: Any,
    device: torch.device,
) -> ModelInputBatch:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has no padding or EOS token")

    max_length = max(len(sample.input_ids) for sample in samples)
    input_ids = torch.full(
        (len(samples), max_length),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sample in enumerate(samples):
        length = len(sample.input_ids)
        input_ids[row, -length:] = torch.tensor(sample.input_ids, dtype=torch.long)
        attention_mask[row, -length:] = 1

    return ModelInputBatch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )


def _convert_to_benchmark_samples(dataset: Path) -> tuple[BenchmarkSample, ...]:
    entries = _SHARE_GPT_ENTRIES.validate_json(dataset.read_bytes())
    return tuple(
        BenchmarkSample(
            prompt=entry.prompt,
            completion=entry.completion,
        )
        for entry in entries
        if len(entry.conversations) >= 2
    )
