"""Regression tests for vLLM model-level routed-expert weight loading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, is_pp_missing_parameter

from expertkit_vllm.experts import remote_moe, remote_routed_experts
from expertkit_vllm.experts.remote_routed_experts import RemoteRoutedExperts


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


class _AutoLoaderHarness(nn.Module):
    def __init__(self, remote_experts: nn.Module, gate: nn.Module, shared: nn.Module) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_DecoderLayer(nn.Identity()), _DecoderLayer(_MoELayer(remote_experts, gate, shared))]
        )


def _make_remote_runner(monkeypatch) -> tuple[remote_moe.RemoteMoERunner, nn.Linear, nn.Linear]:
    compilation = SimpleNamespace(
        static_forward_context={},
        static_all_moe_layers=[],
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=27)),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
        ),
    )
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(
        remote_routed_experts,
        "get_current_vllm_config",
        lambda: vllm_config,
    )
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.setenv("EK_INSTANCE_ID", "7")
    gate = nn.Linear(2, 2, bias=False)
    shared = nn.Linear(2, 2, bias=False)
    moe_config = cast(
        FusedMoEConfig,
        SimpleNamespace(
            num_experts=2,
            num_logical_experts=2,
            experts_per_token=1,
            hidden_dim=2,
            activation=MoEActivation.SILU,
            tp_size=1,
            pcp_size=1,
            is_sequence_parallel=False,
            has_bias=False,
            moe_parallel_config=SimpleNamespace(
                use_ep=False,
                enable_eplb=False,
            ),
        ),
    )
    expert_map_manager = cast(
        ExpertMapManager,
        SimpleNamespace(
            num_fused_shared_experts=0,
            placement_strategy="linear",
        ),
    )
    routed_experts = RemoteRoutedExperts(
        "layers.1.mlp.experts",
        torch.float32,
        moe_config,
        None,
        expert_map_manager,
    )
    runner = remote_moe.RemoteMoERunner(
        layer_name="layers.1.mlp.experts",
        moe_config=moe_config,
        router=cast(FusedMoERouter, nn.Identity()),
        routed_experts=routed_experts,
        gate=gate,
        shared_experts=shared,
        routed_scaling_factor=1.0,
    )
    return runner, gate, shared


def test_deepseek_loader_discards_only_remote_routed_expert_weights(monkeypatch) -> None:
    runner, gate, shared = _make_remote_runner(monkeypatch)
    model = _DeepseekLoaderHarness(runner, gate, shared)

    routed_param = "layers.1.mlp.experts.routed_experts.w2_weight"
    assert not is_pp_missing_parameter(routed_param, model)
    assert not is_pp_missing_parameter("layers.1.mlp.gate.weight", model)
    assert not is_pp_missing_parameter("layers.1.mlp.shared_experts.weight", model)
    assert dict(model.named_parameters())[routed_param].numel() == 0

    gate_weight = torch.full_like(gate.weight, 3.0)
    loaded = DeepseekV2Model.load_weights(
        cast(DeepseekV2Model, model),
        [
            ("layers.1.mlp.experts.0.down_proj.weight", torch.zeros(2, 2)),
            ("layers.1.mlp.gate.weight", gate_weight),
        ],
    )

    assert routed_param in loaded
    assert "layers.1.mlp.gate.weight" in loaded
    assert dict(model.named_parameters())[routed_param].numel() == 0
    torch.testing.assert_close(gate.weight, gate_weight)


def test_runner_discards_auto_loader_expert_subtree(monkeypatch) -> None:
    runner, _, _ = _make_remote_runner(monkeypatch)
    weights = [
        ("0.gate_proj.weight", torch.ones(2, 2)),
        ("0.down_proj.weight", torch.ones(2, 2)),
        ("0.up_proj.weight", torch.ones(2, 2)),
    ]
    consumed: list[str] = []

    def tracked_weights():
        for name, weight in weights:
            consumed.append(name)
            yield name, weight

    assert runner.load_weights(tracked_weights()) == {
        "routed_experts.w13_weight",
        "routed_experts.w2_weight",
    }
    assert consumed == [name for name, _ in weights]
    assert all(parameter.numel() == 0 for parameter in runner.routed_experts.parameters())


def test_auto_loader_tracks_remote_expert_sinks_as_initialized(monkeypatch) -> None:
    runner, gate, shared = _make_remote_runner(monkeypatch)
    model = _AutoLoaderHarness(runner, gate, shared)
    weights = [
        ("model.layers.1.mlp.gate.weight", torch.ones_like(gate.weight)),
        (
            "model.layers.1.mlp.shared_experts.weight",
            torch.ones_like(shared.weight),
        ),
        (
            "model.layers.1.mlp.experts.0.gate_proj.weight",
            torch.ones(2, 2),
        ),
        (
            "model.layers.1.mlp.experts.0.down_proj.weight",
            torch.ones(2, 2),
        ),
        (
            "model.layers.1.mlp.experts.0.up_proj.weight",
            torch.ones(2, 2),
        ),
    ]

    loaded = AutoWeightsLoader(model).load_weights(iter(weights))

    assert loaded == {name for name, _ in model.named_parameters()}
    assert {
        "model.layers.1.mlp.experts.routed_experts.w13_weight",
        "model.layers.1.mlp.experts.routed_experts.w2_weight",
    } <= loaded


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
