"""Regression tests for vLLM model-level routed-expert weight loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.utils import is_pp_missing_parameter

from expertkit_vllm.experts import remote_moe


class _MoELayer(nn.Module):
    def __init__(self, experts: nn.Module, gate: nn.Module, shared_experts: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.shared_experts = shared_experts
        self.experts = experts


class _DecoderLayer(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp


class _DeepseekLoaderHarness(nn.Module):
    def __init__(self, remote_experts: nn.Module, gate: nn.Module, shared: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DecoderLayer(nn.Identity()), _DecoderLayer(_MoELayer(remote_experts, gate, shared))]
        )
        self.config = SimpleNamespace(n_routed_experts=2, n_shared_experts=1)
        self.num_redundant_experts = 0
        self.use_mha = False


def _make_remote_runner(monkeypatch) -> tuple[remote_moe.RemoteMoERunner, nn.Linear, nn.Linear]:
    compilation = SimpleNamespace(
        static_forward_context={},
        static_all_moe_layers=[],
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(
        remote_moe,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=compilation,
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=27)),
        ),
    )
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    gate = nn.Linear(2, 2, bias=False)
    shared = nn.Linear(2, 2, bias=False)
    runner = remote_moe.RemoteMoERunner(
        num_experts=2,
        top_k=1,
        hidden_size=2,
        prefix="layers.1.mlp.experts",
        router=nn.Identity(),
        gate=gate,
        shared_experts=shared,
        apply_routed_scale_to_output=False,
        routed_scaling_factor=1.0,
    )
    return runner, gate, shared


def test_deepseek_loader_skips_only_remote_routed_expert_weights(monkeypatch) -> None:
    runner, gate, shared = _make_remote_runner(monkeypatch)
    model = _DeepseekLoaderHarness(runner, gate, shared)

    routed_param = "layers.1.mlp.experts.routed_experts.w2_weight"
    assert is_pp_missing_parameter(routed_param, model)
    assert not is_pp_missing_parameter("layers.1.mlp.gate.weight", model)
    assert not is_pp_missing_parameter("layers.1.mlp.shared_experts.weight", model)

    gate_weight = torch.full_like(gate.weight, 3.0)
    loaded = DeepseekV2Model.load_weights(
        model,
        [
            ("layers.1.mlp.experts.0.down_proj.weight", torch.zeros(2, 2)),
            ("layers.1.mlp.gate.weight", gate_weight),
        ],
    )

    assert "layers.1.mlp.experts.0.down_proj.weight" in loaded
    assert "layers.1.mlp.gate.weight" in loaded
    torch.testing.assert_close(gate.weight, gate_weight)


def test_layer_name_encoding_is_traceable_by_torch_compile() -> None:
    def encode_during_forward(value: torch.Tensor) -> torch.Tensor:
        remote_moe._encode_layer_name("layers.1.mlp.experts")
        return value + 1

    compiled = torch.compile(encode_during_forward, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(torch.zeros(1)), torch.ones(1))


def test_client_creation_uses_model_metadata_captured_during_layer_init(monkeypatch) -> None:
    runner, _, _ = _make_remote_runner(monkeypatch)
    created: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, endpoint: str, **kwargs) -> None:
            created.append({"endpoint": endpoint, **kwargs})

        def start(self, *, timeout_seconds: float) -> None:
            created[-1]["timeout_seconds"] = timeout_seconds

        def close(self) -> None:
            return None

    def unavailable_config():
        raise AssertionError("vLLM config is unavailable during custom-op execution")

    remote_moe.close_clients()
    monkeypatch.setattr(remote_moe, "BlockingRoutedMoEClient", FakeClient)
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", unavailable_config)

    client = remote_moe._client_for(runner, torch.zeros(1, 2))

    assert isinstance(client, FakeClient)
    assert created[0]["num_layers"] == 27
    remote_moe.close_clients()
