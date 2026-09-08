"""Tests for explicit, platform-cooperative vLLM factory registration."""

import sys
import tomllib
import types
from pathlib import Path
from typing import Any

from expertkit_vllm import plugin


def _module_attribute(module: types.ModuleType, name: str) -> Any:
    return module.__dict__[name]


def install_fake_vllm_modules(monkeypatch, *, device_type: str = "cuda"):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "vllm",
            "vllm.platforms",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.fused_moe",
            "vllm.model_executor.layers.fused_moe.layer",
        )
    }
    package = modules["vllm.model_executor.layers.fused_moe"]
    layer = modules["vllm.model_executor.layers.fused_moe.layer"]

    def platform_fused_moe():
        return "platform"

    package.__dict__["FusedMoE"] = platform_fused_moe
    layer.__dict__["FusedMoE"] = platform_fused_moe

    class FakePlatform:
        def __init__(self) -> None:
            self.device_type = device_type
            self.pre_register_calls = 0

        def pre_register_and_update(self) -> None:
            self.pre_register_calls += 1

            def ascend_fused_moe():
                return "ascend"

            package.__dict__["FusedMoE"] = ascend_fused_moe
            layer.__dict__["FusedMoE"] = ascend_fused_moe

    platform = FakePlatform()
    modules["vllm.platforms"].__dict__["current_platform"] = platform

    remote = types.ModuleType("expertkit_vllm.experts.remote_moe")
    wrapped_factories: set[object] = set()

    def is_expertkit_fused_moe_factory(factory) -> bool:
        return factory in wrapped_factories

    def wrap_fused_moe_factory(factory):
        def wrapped():
            return factory()

        wrapped_factories.add(wrapped)
        return wrapped

    remote.__dict__["is_expertkit_fused_moe_factory"] = is_expertkit_fused_moe_factory
    remote.__dict__["wrap_fused_moe_factory"] = wrap_fused_moe_factory
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "expertkit_vllm.experts.remote_moe", remote)
    return modules, platform


def test_register_wraps_both_vllm_factory_exports(monkeypatch) -> None:
    modules, platform = install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv("EK_ENABLE", "1")

    plugin.register()

    package = modules["vllm.model_executor.layers.fused_moe"]
    layer = modules["vllm.model_executor.layers.fused_moe.layer"]
    package_factory = _module_attribute(package, "FusedMoE")
    layer_factory = _module_attribute(layer, "FusedMoE")
    assert package_factory is layer_factory
    assert package_factory() == "platform"
    assert platform.pre_register_calls == 0


def test_register_captures_ascend_factory_after_platform_pre_registration(
    monkeypatch,
) -> None:
    modules, platform = install_fake_vllm_modules(monkeypatch, device_type="npu")
    monkeypatch.setenv("EK_ENABLE", "1")

    plugin.register()

    package = modules["vllm.model_executor.layers.fused_moe"]
    layer = modules["vllm.model_executor.layers.fused_moe.layer"]
    package_factory = _module_attribute(package, "FusedMoE")
    layer_factory = _module_attribute(layer, "FusedMoE")
    assert package_factory is layer_factory
    assert package_factory() == "ascend"
    assert platform.pre_register_calls == 1


def test_register_is_idempotent(monkeypatch) -> None:
    modules, _ = install_fake_vllm_modules(monkeypatch)
    monkeypatch.setenv("EK_ENABLE", "1")

    plugin.register()
    layer = modules["vllm.model_executor.layers.fused_moe.layer"]
    first = _module_attribute(layer, "FusedMoE")
    plugin.register()

    assert _module_attribute(layer, "FusedMoE") is first


def test_register_is_inert_unless_explicitly_enabled(monkeypatch) -> None:
    modules, platform = install_fake_vllm_modules(monkeypatch)
    monkeypatch.delenv("EK_ENABLE", raising=False)

    plugin.register()

    package = modules["vllm.model_executor.layers.fused_moe"]
    factory = _module_attribute(package, "FusedMoE")
    assert factory() == "platform"
    assert platform.pre_register_calls == 0


def test_pyproject_pins_vllm_and_registers_the_plugin() -> None:
    path = Path(__file__).parents[1] / "pyproject.toml"
    with path.open("rb") as source:
        pyproject = tomllib.load(source)

    assert "vllm==0.25.1" in pyproject["project"]["dependencies"]
    entry_points = pyproject["project"]["entry-points"]["vllm.general_plugins"]
    assert entry_points == {"register_expertkit": "expertkit_vllm.plugin:register"}
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
