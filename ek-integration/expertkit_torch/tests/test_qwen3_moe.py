"""Tests for the Qwen routed-layer integration boundary."""

import importlib
import sys
import types
from typing import ClassVar

import torch


class FakeRoutedClient:
    instances: ClassVar[list["FakeRoutedClient"]] = []

    def __init__(self, endpoint: str, **configuration: object) -> None:
        self.endpoint = endpoint
        self.configuration = configuration
        self.call: dict[str, object] | None = None
        self.__class__.instances.append(self)

    def forward_layer(self, **call: object) -> torch.Tensor:
        self.call = call
        hidden_states = call["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        return torch.full_like(hidden_states, 5)


def import_qwen_module(monkeypatch):
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    transformers.AutoModelForCausalLM = object
    utils = types.ModuleType("transformers.utils")
    logging = types.ModuleType("transformers.utils.logging")
    logging.set_verbosity_error = lambda: None
    models = types.ModuleType("transformers.models")
    qwen_package = types.ModuleType("transformers.models.qwen3_moe")
    modeling = types.ModuleType("transformers.models.qwen3_moe.modeling_qwen3_moe")
    modeling.Qwen3MoeSparseMoeBlock = object
    modeling.Qwen3MoeMLP = object
    qwen_package.modeling_qwen3_moe = modeling

    for name, module in (
        ("transformers", transformers),
        ("transformers.utils", utils),
        ("transformers.utils.logging", logging),
        ("transformers.models", models),
        ("transformers.models.qwen3_moe", qwen_package),
        ("transformers.models.qwen3_moe.modeling_qwen3_moe", modeling),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("expertkit_torch.models.qwen3_moe", None)
    module = importlib.import_module("expertkit_torch.models.qwen3_moe")
    monkeypatch.setattr(module, "RoutedMoEClient", FakeRoutedClient)
    return module, modeling


class QwenConfig:
    num_hidden_layers = 1
    num_experts = 3
    num_experts_per_tok = 2
    hidden_size = 2
    moe_intermediate_size = 4
    norm_topk_prob = True


def test_remote_qwen_path_sends_final_fp32_routing_and_uses_aggregated_result(
    monkeypatch,
) -> None:
    FakeRoutedClient.instances.clear()
    module, modeling = import_qwen_module(monkeypatch)
    module.intercept_moe(
        enable_ek=True,
        ek_addr="127.0.0.1:50050",
        ek_instance_id=7,
    )
    layer = modeling.Qwen3MoeSparseMoeBlock(QwenConfig())
    with torch.no_grad():
        layer.gate.weight.copy_(torch.tensor([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))

    output, router_logits = layer(torch.tensor([[[1.0, 0.0]]], dtype=torch.float32))

    torch.testing.assert_close(output, torch.tensor([[[5.0, 5.0]]]))
    assert router_logits.shape == (1, 3)
    fake = FakeRoutedClient.instances[0]
    assert fake.configuration["instance_id"] == 7
    assert fake.call is not None
    expert_ids = fake.call["expert_ids"]
    routing_weights = fake.call["routing_weights"]
    assert isinstance(expert_ids, torch.Tensor)
    assert isinstance(routing_weights, torch.Tensor)
    assert expert_ids.tolist() == [[0, 1]]
    assert routing_weights.dtype is torch.float32
    torch.testing.assert_close(routing_weights.sum(dim=1), torch.ones(1))
