"""Semantic tests shared by the supported routed MoE model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from expertkit_torch.models._common import RoutedLayerIds
from expertkit_torch.models.deepseek_v2 import create_routed_moe_class as deepseek_v2_class
from expertkit_torch.models.deepseek_v3 import create_routed_moe_class as deepseek_v3_class
from expertkit_torch.models.loader import _deepseek_layer_ids, _qwen_layer_ids
from expertkit_torch.models.mixtral import create_routed_moe_class as mixtral_class
from expertkit_torch.models.qwen3_moe import create_routed_moe_class as qwen3_class


class LocalExpertClient:
    """Evaluate packed reference expert weights behind the client interface."""

    def __init__(self, experts: nn.Module) -> None:
        self.experts = experts
        self.calls: list[dict[str, Any]] = []

    def forward_layer(self, **call: Any) -> torch.Tensor:
        self.calls.append(call)
        hidden_states = call["hidden_states"]
        expert_ids = call["expert_ids"]
        routing_weights = call["routing_weights"]
        result = torch.zeros_like(hidden_states, dtype=torch.float32)
        for token_index in range(hidden_states.shape[0]):
            for assignment_index in range(expert_ids.shape[1]):
                expert_id = int(expert_ids[token_index, assignment_index])
                token = hidden_states[token_index : token_index + 1]
                gate, up = F.linear(token, self.experts.gate_up_proj[expert_id]).chunk(
                    2,
                    dim=-1,
                )
                activated = self.experts.act_fn(gate) * up
                expert_output = F.linear(
                    activated,
                    self.experts.down_proj[expert_id],
                )[0]
                result[token_index] += (
                    expert_output.float() * routing_weights[token_index, assignment_index]
                )
        return result.to(hidden_states.dtype)


@dataclass(frozen=True)
class AdapterCase:
    name: str
    config: Any
    native_class: type[nn.Module]
    routed_class: Any


CASES = (
    AdapterCase(
        "qwen3_moe",
        Qwen3MoeConfig(
            hidden_size=4,
            intermediate_size=8,
            moe_intermediate_size=6,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        ),
        Qwen3MoeSparseMoeBlock,
        qwen3_class,
    ),
    AdapterCase(
        "deepseek_v2",
        DeepseekV2Config(
            hidden_size=4,
            intermediate_size=8,
            moe_intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            first_k_dense_replace=1,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
        ),
        DeepseekV2Moe,
        deepseek_v2_class,
    ),
    AdapterCase(
        "deepseek_v3",
        DeepseekV3Config(
            hidden_size=4,
            intermediate_size=8,
            moe_intermediate_size=6,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            first_k_dense_replace=1,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
        ),
        DeepseekV3MoE,
        deepseek_v3_class,
    ),
    AdapterCase(
        "mixtral",
        MixtralConfig(
            hidden_size=4,
            intermediate_size=6,
            num_hidden_layers=1,
            num_local_experts=4,
            num_experts_per_tok=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        MixtralSparseMoeBlock,
        mixtral_class,
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_routed_adapter_matches_native_moe_and_sends_final_routing(
    case: AdapterCase,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(11)
    native = case.native_class(case.config)
    for parameter in native.parameters():
        nn.init.normal_(parameter, mean=0.0, std=0.1)
    native = native.to(dtype).eval()
    client = LocalExpertClient(native.experts)
    routed_type = case.routed_class(client, RoutedLayerIds((7,)))
    routed = routed_type(case.config).to(dtype).eval()
    non_expert_state = {
        name: value
        for name, value in native.state_dict().items()
        if not name.startswith("experts.")
    }
    routed.load_state_dict(non_expert_state, strict=True)
    hidden_states = torch.randn(2, 3, case.config.hidden_size, dtype=dtype)

    with torch.no_grad():
        expected = native(hidden_states.clone())
        actual = routed(hidden_states.clone())

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    tolerance = 0.02 if dtype is torch.bfloat16 else 1e-6
    torch.testing.assert_close(
        actual,
        expected,
        atol=tolerance,
        rtol=tolerance,
    )
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["layer_id"] == 7
    assert call["hidden_states"].shape == (6, case.config.hidden_size)
    assert call["expert_ids"].shape == (6, case.config.num_experts_per_tok)
    assert call["routing_weights"].dtype is torch.float32
    flattened = hidden_states.reshape(-1, case.config.hidden_size)
    if case.name in ("qwen3_moe", "mixtral"):
        _, expected_weights, expected_ids = native.gate(flattened)
    elif case.name == "deepseek_v2":
        router_logits = F.linear(
            hidden_states.float(),
            native.gate.weight.float(),
        )
        expected_ids, expected_weights = native.route_tokens_to_experts(router_logits)
    else:
        expected_ids, expected_weights = native.route_tokens_to_experts(native.gate(hidden_states))
    torch.testing.assert_close(call["expert_ids"], expected_ids)
    torch.testing.assert_close(call["routing_weights"], expected_weights)


def test_layer_id_mapping_preserves_sparse_and_dense_model_layers() -> None:
    qwen_config = Qwen3MoeConfig(
        num_hidden_layers=6,
        num_experts=4,
        decoder_sparse_step=2,
        mlp_only_layers=[3],
    )
    deepseek_config = DeepseekV2Config(
        num_hidden_layers=5,
        first_k_dense_replace=2,
        num_experts_per_tok=2,
    )

    assert _qwen_layer_ids(qwen_config) == (1, 5)
    assert _deepseek_layer_ids(deepseek_config) == (2, 3, 4)


def test_layer_id_source_rejects_missing_or_extra_blocks() -> None:
    ids = RoutedLayerIds((2,))
    with pytest.raises(RuntimeError, match="expected"):
        ids.require_complete()
    assert ids.take() == 2
    ids.require_complete()
    with pytest.raises(RuntimeError, match="more routed"):
        ids.take()
