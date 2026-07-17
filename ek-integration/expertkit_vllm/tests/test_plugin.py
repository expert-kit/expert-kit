"""Tests for explicit vLLM factory registration and version targeting."""

import ast
import sys
import types
from pathlib import Path

from expertkit_vllm import plugin


def install_fake_vllm_modules(monkeypatch):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.fused_moe",
            "vllm.model_executor.layers.fused_moe.layer",
        )
    }
    remote = types.ModuleType("expertkit_vllm.experts.remote_moe")

    def remote_fused_moe():
        return None

    remote.remote_fused_moe = remote_fused_moe
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "expertkit_vllm.experts.remote_moe", remote)
    return modules, remote_fused_moe


def test_register_replaces_both_vllm_factory_exports(monkeypatch) -> None:
    modules, replacement = install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv("EK_ENABLE", "1")

    plugin.register()

    package = modules["vllm.model_executor.layers.fused_moe"]
    layer = modules["vllm.model_executor.layers.fused_moe.layer"]
    assert package.FusedMoE is replacement
    assert layer.FusedMoE is replacement


def test_register_is_inert_unless_explicitly_enabled(monkeypatch) -> None:
    modules, _ = install_fake_vllm_modules(monkeypatch)
    monkeypatch.delenv("EK_ENABLE", raising=False)

    plugin.register()

    assert not hasattr(modules["vllm.model_executor.layers.fused_moe"], "FusedMoE")


def test_remote_factory_matches_the_vllm_0251_parameter_surface() -> None:
    source = Path(__file__).parents[1] / "expertkit_vllm" / "experts" / "remote_moe.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "remote_fused_moe"
    )
    parameters = [argument.arg for argument in function.args.args]
    assert parameters == [
        "num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
        "intermediate_pad",
        "params_dtype",
        "renormalize",
        "use_grouped_topk",
        "num_expert_group",
        "topk_group",
        "quant_config",
        "tp_size",
        "dp_size",
        "pcp_size",
        "prefix",
        "custom_routing_function",
        "router",
        "scoring_func",
        "routed_scaling_factor",
        "swiglu_limit",
        "swiglu_alpha",
        "swiglu_beta",
        "e_score_correction_bias",
        "apply_router_weight_on_input",
        "activation",
        "enable_eplb",
        "num_redundant_experts",
        "has_bias",
        "is_sequence_parallel",
        "reduce_results",
        "ckpt_names",
        "n_shared_experts",
        "router_logits_dtype",
        "gate",
        "shared_experts",
        "shared_expert_gate",
        "routed_input_transform",
        "routed_output_transform",
        "apply_routed_scale_to_output",
        "zero_expert_type",
        "hash_indices_table",
        "runner_cls",
        "runner_args",
        "routed_experts_cls",
        "routed_experts_args",
    ]


def test_setup_pins_the_reviewed_vllm_version() -> None:
    setup = (Path(__file__).parents[1] / "setup.py").read_text(encoding="utf-8")
    assert '"vllm==0.25.1"' in setup
