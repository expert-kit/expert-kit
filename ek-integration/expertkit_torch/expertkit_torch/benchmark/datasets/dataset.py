from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    prompt: str
    completion: str


@dataclass(frozen=True, slots=True)
class ModelInputBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def input_tokens(self) -> int:
        return int(self.attention_mask.sum().item())


class BenchmarkDataset(Protocol):
    def iter_batches(
        self,
        tokenizer: Any,
        *,
        batch_size: int,
        num_prompts: int,
        output_length: int,
        device: torch.device,
    ) -> Iterator[ModelInputBatch]: ...
