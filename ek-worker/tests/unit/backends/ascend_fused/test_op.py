# ruff: noqa: E402

import sys
from types import ModuleType

import pytest
import torch

sys.modules.setdefault("torch_npu", ModuleType("torch_npu"))

from expertkit_worker.backends.ascend_fused import op


def test_dynamic_w8a8_passes_list_weights_and_scales_to_gmm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def route(hidden: torch.Tensor, _ids: torch.Tensor, **kwargs: object):
        calls.append({"operator": "route", **kwargs})
        return (
            hidden.to(torch.int8),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            torch.ones(hidden.shape[0], dtype=torch.float32),
        )

    def grouped_matmul(**kwargs: object):
        calls.append({"operator": "gmm", **kwargs})
        x = kwargs["x"]
        weight = kwargs["weight"]
        assert isinstance(x, list) and isinstance(weight, list)
        return [torch.ones((x[0].shape[0], weight[0].shape[1]), dtype=torch.bfloat16)]

    def dynamic_quant(x: torch.Tensor, **_kwargs: object):
        return x.to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)

    def unpermute(**kwargs: object):
        calls.append({"operator": "unpermute", **kwargs})
        return kwargs["permuted_tokens"]

    monkeypatch.setattr(op.torch_npu, "npu_moe_init_routing_v2", route, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_grouped_matmul", grouped_matmul, raising=False)
    monkeypatch.setattr(
        op.torch_npu,
        "npu_swiglu",
        lambda x, dim: x.chunk(2, dim=dim)[0],
        raising=False,
    )
    monkeypatch.setattr(op.torch_npu, "npu_dynamic_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(op.torch_npu, "npu_moe_token_unpermute", unpermute, raising=False)

    w13 = [torch.ones((2, 4), dtype=torch.int8) for _ in range(2)]
    w2 = [torch.ones((2, 2), dtype=torch.int8) for _ in range(2)]
    w13_scale = [torch.ones(4, dtype=torch.bfloat16) for _ in range(2)]
    w2_scale = [torch.ones(2, dtype=torch.bfloat16) for _ in range(2)]
    output = op.run_ascend_fused_moe(
        torch.ones((2, 2), dtype=torch.bfloat16),
        torch.tensor([[0], [1]], dtype=torch.int32),
        torch.ones((2, 1), dtype=torch.float32),
        w13=w13,
        w2=w2,
        compute_type=op.AscendFusedComputeType.DYNAMIC_W8A8,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )

    route_call = calls[0]
    first_gmm, second_gmm = (call for call in calls if call["operator"] == "gmm")
    assert route_call["quant_mode"] == 1
    assert first_gmm["weight"] is w13
    assert first_gmm["scale"] is w13_scale
    assert second_gmm["weight"] is w2
    assert second_gmm["scale"] is w2_scale
    assert all(scale.dtype is torch.bfloat16 for scale in w13_scale + w2_scale)
    assert first_gmm["output_dtype"] is torch.bfloat16
    assert second_gmm["output_dtype"] is torch.bfloat16
    assert output.shape == (2, 2)


def test_dynamic_w8a8_requires_both_scale_lists() -> None:
    with pytest.raises(ValueError, match="requires w13_scale and w2_scale"):
        op.run_ascend_fused_moe(
            torch.ones((1, 2), dtype=torch.bfloat16),
            torch.zeros((1, 1), dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.float32),
            w13=[torch.ones((2, 4), dtype=torch.int8)],
            w2=[torch.ones((2, 2), dtype=torch.int8)],
            compute_type=op.AscendFusedComputeType.DYNAMIC_W8A8,
        )
