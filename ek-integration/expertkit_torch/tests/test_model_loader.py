"""Tests for bounded Transformers class replacement during model loading."""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pytest
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from expertkit_torch.models import loader
from expertkit_torch.models._common import RoutedLayerIds


class FakeClient:
    instances: ClassVar[list[FakeClient]] = []

    def __init__(self, endpoint: str, **configuration: Any) -> None:
        self.endpoint = endpoint
        self.configuration = configuration
        self.closed = False
        self.__class__.instances.append(self)

    def close(self) -> None:
        self.closed = True


class FakeModel:
    def __init__(self) -> None:
        self.device: object | None = None
        self.evaluation = False

    def to(self, device: object) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.evaluation = True
        return self


def qwen_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=6,
        num_hidden_layers=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
    )


def install_loading_fakes(monkeypatch: pytest.MonkeyPatch, config: Qwen3MoeConfig) -> None:
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda _: config)
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda _: object())
    monkeypatch.setattr(loader, "RoutedMoEClient", FakeClient)


def test_expertkit_load_uses_real_layer_ids_and_restores_transformers_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.instances.clear()
    config = qwen_config()
    install_loading_fakes(monkeypatch, config)
    original = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
    causal_lm_class = AutoModelForCausalLM._model_mapping[type(config)]
    original_ignored_keys = causal_lm_class.__dict__.get(
        "_keys_to_ignore_on_load_unexpected",
        loader._MISSING,
    )
    constructed_layer_ids: list[int] = []

    def from_pretrained(_: str, *, dtype: object) -> FakeModel:
        assert dtype == "auto"
        assert loader._ROUTED_EXPERT_WEIGHT_PATTERN in (
            causal_lm_class._keys_to_ignore_on_load_unexpected
        )
        replacement = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock
        assert replacement is not original
        constructed_layer_ids.extend([replacement(config).layer_id, replacement(config).layer_id])
        return FakeModel()

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", from_pretrained)

    loaded = loader.load_model(
        "/models/qwen",
        controller_endpoint="127.0.0.1:5002",
        device="cpu",
    )

    assert modeling_qwen3_moe.Qwen3MoeSparseMoeBlock is original
    assert (
        causal_lm_class.__dict__.get(
            "_keys_to_ignore_on_load_unexpected",
            loader._MISSING,
        )
        is original_ignored_keys
    )
    assert constructed_layer_ids == [1, 3]
    assert loaded.model.device.type == "cpu"
    assert loaded.model.evaluation
    client = FakeClient.instances[0]
    assert client.configuration == {
        "instance_id": None,
        "num_layers": 4,
        "experts_per_layer": 4,
        "hidden_dim": 4,
        "top_k": 2,
    }
    loaded.close()
    loaded.close()
    assert client.closed


def test_failed_load_restores_class_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.instances.clear()
    config = qwen_config()
    install_loading_fakes(monkeypatch, config)
    original = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock

    def fail(_: str, *, dtype: object) -> FakeModel:
        assert dtype == "auto"
        assert modeling_qwen3_moe.Qwen3MoeSparseMoeBlock is not original
        raise OSError("broken checkpoint")

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", fail)

    with pytest.raises(OSError, match="broken checkpoint"):
        loader.load_model("/models/qwen", device="cpu")

    assert modeling_qwen3_moe.Qwen3MoeSparseMoeBlock is original
    assert FakeClient.instances[0].closed


def test_local_load_keeps_native_transformers_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.instances.clear()
    config = qwen_config()
    install_loading_fakes(monkeypatch, config)
    original = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock

    def from_pretrained(_: str, *, dtype: object) -> FakeModel:
        assert dtype == "auto"
        assert modeling_qwen3_moe.Qwen3MoeSparseMoeBlock is original
        return FakeModel()

    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", from_pretrained)

    with loader.load_model("/models/qwen", mode="local", device="cpu") as loaded:
        assert loaded.model_type == "qwen3_moe"

    assert not FakeClient.instances


@pytest.mark.parametrize(
    ("config", "expected_layer_ids"),
    [
        (
            Qwen3MoeConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                moe_intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_experts=4,
                num_experts_per_tok=2,
            ),
            [0, 1],
        ),
        (
            DeepseekV2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                moe_intermediate_size=8,
                num_hidden_layers=3,
                first_k_dense_replace=1,
                n_routed_experts=4,
                n_shared_experts=1,
                num_experts_per_tok=2,
                n_group=1,
                topk_group=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                kv_lora_rank=4,
                q_lora_rank=None,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
            ),
            [1, 2],
        ),
        (
            DeepseekV3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                moe_intermediate_size=8,
                num_hidden_layers=3,
                first_k_dense_replace=1,
                n_routed_experts=4,
                n_shared_experts=1,
                num_experts_per_tok=2,
                n_group=2,
                topk_group=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                kv_lora_rank=4,
                q_lora_rank=None,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
            ),
            [1, 2],
        ),
        (
            MixtralConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=2,
                num_local_experts=4,
                num_experts_per_tok=2,
                num_attention_heads=2,
                num_key_value_heads=1,
            ),
            [0, 1],
        ),
    ],
    ids=("qwen3_moe", "deepseek_v2", "deepseek_v3", "mixtral"),
)
def test_transformers_constructs_replacement_blocks_at_real_layer_ids(
    config: Any,
    expected_layer_ids: list[int],
) -> None:
    spec = loader._adapter_spec(config.model_type)
    layer_ids = RoutedLayerIds(spec.layer_ids(config))
    replacement = spec.create_class(FakeClient("unused"), layer_ids)
    original = getattr(spec.module, spec.class_name)

    with loader._temporary_model_class(spec, replacement):
        model = AutoModelForCausalLM.from_config(config)

    assert getattr(spec.module, spec.class_name) is original
    layer_ids.require_complete()
    blocks = [layer.mlp for layer in model.model.layers if hasattr(layer.mlp, "layer_id")]
    assert [block.layer_id for block in blocks] == expected_layer_ids


@pytest.mark.parametrize(
    "checkpoint_key",
    [
        "model.layers.2.mlp.experts.7.gate_proj.weight",
        "model.layers.2.mlp.experts.7.up_proj.weight",
        "model.layers.2.mlp.experts.7.down_proj.weight",
        "model.layers.2.mlp.experts.gate_up_proj",
        "model.layers.2.mlp.experts.gate_up_proj.weight",
        "model.layers.2.mlp.experts.down_proj",
        "model.layers.2.block_sparse_moe.experts.gate_up_proj",
    ],
)
def test_routed_expert_weight_pattern_covers_old_and_packed_keys(
    checkpoint_key: str,
) -> None:
    assert re.search(loader._ROUTED_EXPERT_WEIGHT_PATTERN, checkpoint_key)


@pytest.mark.parametrize(
    "checkpoint_key",
    [
        "model.layers.2.mlp.gate.weight",
        "model.layers.2.mlp.shared_experts.gate_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
    ],
)
def test_routed_expert_weight_pattern_keeps_frontend_keys(
    checkpoint_key: str,
) -> None:
    assert re.search(loader._ROUTED_EXPERT_WEIGHT_PATTERN, checkpoint_key) is None


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"model_path": ""}, "model_path"),
        ({"model_path": "/m", "mode": "remote"}, "mode"),
        ({"model_path": "/m", "controller_endpoint": ""}, "controller_endpoint"),
        ({"model_path": "/m", "instance_id": 0}, "instance_id"),
        ({"model_path": "/m", "dtype": "int8"}, "dtype"),
    ],
)
def test_load_model_rejects_invalid_arguments(
    arguments: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        loader.load_model(**arguments)
