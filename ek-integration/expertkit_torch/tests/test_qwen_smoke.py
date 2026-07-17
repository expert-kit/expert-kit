"""Tests for the fixed Qwen generation check."""

from __future__ import annotations

import sys
import types

import pytest

from expertkit_torch.qwen_smoke import run_qwen_smoke


def install_fake_evaluator(monkeypatch: pytest.MonkeyPatch, payload: object) -> list[object]:
    calls: list[object] = []
    module = types.ModuleType("expertkit_torch.models.qwen3_moe")

    def evaluate_batch(**arguments: object) -> object:
        calls.append(arguments)
        return payload

    module.evaluate_batch = evaluate_batch
    monkeypatch.setitem(sys.modules, "expertkit_torch.models.qwen3_moe", module)
    return calls


def test_qwen_smoke_fixes_batch_and_generation_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_evaluator(
        monkeypatch,
        {
            "results": [
                {
                    "thinking_content": "short reasoning",
                    "content": "answer",
                    "output_tokens": 20,
                }
            ]
        },
    )

    result = run_qwen_smoke(
        model_path="/models/qwen",
        controller_endpoint="127.0.0.1:5002",
        instance_id=7,
        prompt="hello",
    )

    assert result["content"] == "answer"
    assert calls == [
        {
            "model_path": "/models/qwen",
            "prompts": ["hello"],
            "output_max_length": 20,
            "enable_ek": True,
            "ek_addr": "127.0.0.1:5002",
            "ek_instance_id": 7,
        }
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"results": []}, "exactly one"),
        ({"results": [{"content": "answer", "output_tokens": 0}]}, "between 1 and 20"),
        ({"results": [{"content": "answer", "output_tokens": 21}]}, "between 1 and 20"),
        (
            {"results": [{"thinking_content": "", "content": "", "output_tokens": 1}]},
            "no readable text",
        ),
    ],
)
def test_qwen_smoke_rejects_invalid_generation(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    message: str,
) -> None:
    install_fake_evaluator(monkeypatch, payload)

    with pytest.raises(RuntimeError, match=message):
        run_qwen_smoke(
            model_path="/models/qwen",
            controller_endpoint="127.0.0.1:5002",
            instance_id=7,
        )
