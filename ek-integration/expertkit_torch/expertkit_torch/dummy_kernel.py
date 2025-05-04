from typing import Tuple
import torch


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "This is a dummy kernel. Please use the actual implementation."
    )


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    raise NotImplementedError(
        "This is a dummy kernel. Please use the actual implementation."
    )


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    raise NotImplementedError(
        "This is a dummy kernel. Please use the actual implementation."
    )
