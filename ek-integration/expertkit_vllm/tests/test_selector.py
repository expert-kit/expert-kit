"""Tests for typed default and lazy Ascend expert-selection adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import torch
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)

from expertkit_vllm.experts import selector
from expertkit_vllm.experts.selector import ExpertSelectionOptions


def _options() -> ExpertSelectionOptions:
    return cast(
        ExpertSelectionOptions,
        SimpleNamespace(
            moe_config=cast(
                FusedMoEConfig,
                SimpleNamespace(
                    num_logical_experts=8,
                    num_experts=8,
                ),
            ),
            top_k=2,
            use_grouped_topk=True,
            renormalize=True,
            topk_group=2,
            num_expert_group=4,
            custom_routing_function=None,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
            e_score_correction_bias=torch.ones(8),
        ),
    )


def test_default_selector_delegates_to_the_vllm_router() -> None:
    calls: list[dict[str, object]] = []
    weights = torch.tensor([[0.75, 0.25]])
    ids = torch.tensor([[1, 3]], dtype=torch.int32)

    class FakeRouter:
        def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            topk_indices_dtype: torch.dtype,
            *,
            input_ids: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "router_logits": router_logits,
                    "topk_indices_dtype": topk_indices_dtype,
                    "input_ids": input_ids,
                }
            )
            return weights, ids

    hidden_states = torch.zeros(1, 4)
    router_logits = torch.zeros(1, 8)
    input_ids = torch.tensor([11])
    adapter = selector.VllmExpertSelector(cast(FusedMoERouter, FakeRouter()))

    result = adapter.select_experts(
        hidden_states,
        router_logits,
        input_ids=input_ids,
    )

    assert result == (weights, ids)
    assert len(calls) == 1
    assert calls[0]["hidden_states"] is hidden_states
    assert calls[0]["router_logits"] is router_logits
    assert calls[0]["topk_indices_dtype"] is torch.int32
    assert calls[0]["input_ids"] is input_ids


def test_ascend_selector_loads_lazily_and_forwards_routing_options(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    weights = torch.tensor([[0.6, 0.4]])
    ids = torch.tensor([[2, 5]], dtype=torch.int32)

    def ascend_select_experts(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kwargs)
        return weights, ids

    monkeypatch.setattr(
        selector,
        "import_module",
        lambda name: SimpleNamespace(select_experts=ascend_select_experts),
    )
    tid2eid = torch.arange(8)
    adapter = selector.AscendExpertSelector(_options(), tid2eid=tid2eid)
    hidden_states = torch.zeros(1, 4)
    router_logits = torch.zeros(1, 8)
    input_ids = torch.tensor([13])

    result = adapter.select_experts(
        hidden_states,
        router_logits,
        input_ids=input_ids,
    )

    assert result == (weights, ids)
    assert len(calls) == 1
    call = calls[0]
    assert call["hidden_states"] is hidden_states
    assert call["router_logits"] is router_logits
    assert call["top_k"] == 2
    assert call["use_grouped_topk"] is True
    assert call["renormalize"] is True
    assert call["topk_group"] == 2
    assert call["num_expert_group"] == 4
    assert call["custom_routing_function"] is None
    assert call["scoring_func"] == "sigmoid"
    assert call["routed_scaling_factor"] == 2.5
    torch.testing.assert_close(
        cast(torch.Tensor, call["e_score_correction_bias"]),
        torch.ones(8),
    )
    assert call["indices_type"] is torch.int32
    assert call["mix_placement"] is False
    assert call["num_logical_experts"] == 8
    assert call["num_shared_experts"] == 0
    assert call["num_experts"] == 8
    assert call["input_ids"] is input_ids
    assert call["tid2eid"] is tid2eid


def test_factory_uses_ascend_only_for_the_npu_platform(monkeypatch) -> None:
    router = cast(FusedMoERouter, object())
    options = _options()
    sentinel = cast(selector.ExpertSelector, object())
    monkeypatch.setattr(selector, "AscendExpertSelector", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        selector,
        "current_platform",
        SimpleNamespace(device_type="npu"),
    )

    assert selector.create_expert_selector(router, options) is sentinel
