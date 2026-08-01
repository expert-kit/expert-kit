"""Verify Expert Kit composes with, rather than replaces, a platform factory."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)

from expertkit_vllm.experts import remote_moe, remote_routed_experts
from expertkit_vllm.experts.remote_moe import (
    RemoteMoERunner,
    is_expertkit_fused_moe_factory,
    wrap_fused_moe_factory,
)
from expertkit_vllm.experts.remote_routed_experts import RemoteRoutedExperts


def _vllm_factory_config(
    *,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
) -> SimpleNamespace:
    compilation = SimpleNamespace(
        static_forward_context={},
        static_all_moe_layers=[],
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
        max_cudagraph_capture_size=0,
    )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            tensor_parallel_size=tensor_parallel_size,
            prefill_context_parallel_size=1,
            expert_placement_strategy="linear",
            enable_expert_parallel=enable_expert_parallel,
            enable_eplb=False,
            enable_dbo=False,
            all2all_backend="allgather_reducescatter",
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
        ),
        lora_config=None,
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=compilation,
    )


def _patch_vllm_config(monkeypatch, config: SimpleNamespace) -> None:
    monkeypatch.setattr(fused_moe_layer, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(remote_moe, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(remote_routed_experts, "get_current_vllm_config", lambda: config)
    monkeypatch.setenv("EK_ADDR", "127.0.0.1:50050")
    monkeypatch.delenv("EK_INSTANCE_ID", raising=False)


def _remote_factory_kwargs(prefix: str = "layers.0.mlp.experts") -> dict[str, object]:
    class FakeRouter:
        routing_method_type = RoutingMethodType.Default

    return {
        "num_experts": 2,
        "top_k": 1,
        "hidden_size": 4,
        "intermediate_size": 8,
        "params_dtype": torch.float32,
        "prefix": prefix,
        "router": cast(FusedMoERouter, FakeRouter()),
    }


def test_wrapper_preserves_platform_factory_and_injects_remote_classes() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = cast(MoERunnerInterface, object())

    def platform_factory(
        num_experts: int,
        *,
        tid2eid: object | None = None,
        **kwargs: object,
    ) -> MoERunnerInterface:
        calls.append(((num_experts,), {"tid2eid": tid2eid, **kwargs}))
        return sentinel

    wrapped = wrap_fused_moe_factory(platform_factory)
    tid2eid = object()

    result = wrapped(8, tid2eid=tid2eid)

    assert result is sentinel
    assert calls == [
        (
            (8,),
            {
                "tid2eid": tid2eid,
                "runner_cls": RemoteMoERunner,
                "routed_experts_cls": RemoteRoutedExperts,
                "dp_size": 1,
                "tp_size": 1,
                "pcp_size": 1,
            },
        )
    ]
    assert inspect.signature(wrapped) == inspect.signature(platform_factory)
    assert is_expertkit_fused_moe_factory(wrapped)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("runner_cls", object(), "custom_runner"),
        ("runner_args", {"custom": True}, "custom_runner"),
        ("routed_experts_cls", object(), "custom_experts"),
        ("routed_experts_args", {"custom": True}, "custom_experts"),
    ],
)
def test_wrapper_rejects_conflicting_factory_customization(
    name: str,
    value: object,
    message: str,
) -> None:
    def platform_factory(**kwargs: object) -> MoERunnerInterface:
        raise AssertionError(f"platform factory should not run: {kwargs}")

    wrapped = wrap_fused_moe_factory(platform_factory)

    with pytest.raises(ValueError, match=message):
        cast(Callable[..., MoERunnerInterface], wrapped)(**{name: value})


def test_wrapper_is_idempotent() -> None:
    def platform_factory() -> MoERunnerInterface:
        return cast(MoERunnerInterface, object())

    wrapped = wrap_fused_moe_factory(platform_factory)

    assert wrap_fused_moe_factory(wrapped) is wrapped


def test_vllm_factory_constructs_the_remote_runner_without_local_weights(
    monkeypatch,
) -> None:
    config = _vllm_factory_config()
    _patch_vllm_config(monkeypatch, config)

    wrapped = wrap_fused_moe_factory(fused_moe_layer.FusedMoE)
    runner = cast(Callable[..., MoERunnerInterface], wrapped)(**_remote_factory_kwargs())

    assert isinstance(runner, RemoteMoERunner)
    assert isinstance(runner.routed_experts, RemoteRoutedExperts)
    assert config.parallel_config.data_parallel_size == 2
    assert config.parallel_config.tensor_parallel_size == 1
    assert runner.moe_config.dp_size == 1
    assert runner.moe_config.tp_size == 1
    assert runner.moe_config.pcp_size == 1
    sink_parameters = dict(runner.routed_experts.named_parameters())
    assert set(sink_parameters) == {"w13_weight", "w2_weight"}
    assert all(parameter.numel() == 0 for parameter in sink_parameters.values())
    assert config.compilation_config.static_forward_context == {
        "layers.0.mlp.experts": runner,
    }


def test_vllm_factory_rejects_native_expert_parallel_under_global_dp(
    monkeypatch,
) -> None:
    config = _vllm_factory_config(enable_expert_parallel=True)
    _patch_vllm_config(monkeypatch, config)
    wrapped = wrap_fused_moe_factory(fused_moe_layer.FusedMoE)

    with pytest.raises(ValueError, match="expert_parallel"):
        cast(Callable[..., MoERunnerInterface], wrapped)(**_remote_factory_kwargs())


def test_vllm_factory_rejects_native_tensor_parallel_under_global_dp(
    monkeypatch,
) -> None:
    config = _vllm_factory_config(tensor_parallel_size=2)
    _patch_vllm_config(monkeypatch, config)
    wrapped = wrap_fused_moe_factory(fused_moe_layer.FusedMoE)

    with pytest.raises(ValueError, match="tensor_parallel"):
        cast(Callable[..., MoERunnerInterface], wrapped)(**_remote_factory_kwargs())
