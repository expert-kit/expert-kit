"""Tests for prepared output allocation and validation."""

import pytest
import torch

from expertkit_transport.buffers import TorchOutputBufferProvider
from expertkit_transport.contracts import OutputSpec, PreparedOutput


class ForeignOutput(PreparedOutput):
    @property
    def tensor(self) -> torch.Tensor:
        return torch.empty((2, 4), dtype=torch.float16)


class NoncontiguousOutput(PreparedOutput):
    @property
    def tensor(self) -> torch.Tensor:
        return torch.empty((4, 8), dtype=torch.float16).T


def test_torch_provider_prepares_exact_maximum_output() -> None:
    provider = TorchOutputBufferProvider()
    spec = OutputSpec(8, 4, torch.bfloat16, "cpu")

    output = provider.prepare(spec)

    assert output.tensor.shape == (8, 4)
    assert output.tensor.dtype == torch.bfloat16
    assert output.tensor.device == torch.device("cpu")
    assert output.tensor.is_contiguous()
    provider.validate(output, spec)
    provider.before_receive(output)
    provider.after_consume(output)
    provider.release(output)


@pytest.mark.parametrize(
    "spec",
    [
        lambda: OutputSpec(0, 4, torch.float16, "cpu"),
        lambda: OutputSpec(8, 0, torch.float16, "cpu"),
        lambda: OutputSpec(8, 4, torch.int8, "cpu"),
    ],
)
def test_output_spec_rejects_invalid_capacity(spec: object) -> None:
    with pytest.raises(ValueError):
        spec()  # type: ignore[operator]


def test_torch_provider_rejects_foreign_output_on_release() -> None:
    with pytest.raises(TypeError, match="not prepared"):
        TorchOutputBufferProvider().release(ForeignOutput())


def test_torch_provider_validates_shape_and_layout() -> None:
    provider = TorchOutputBufferProvider()

    with pytest.raises(ValueError, match="wrong shape"):
        provider.validate(ForeignOutput(), OutputSpec(8, 4, torch.float16, "cpu"))
    with pytest.raises(ValueError, match="contiguous"):
        provider.validate(NoncontiguousOutput(), OutputSpec(8, 4, torch.float16, "cpu"))
