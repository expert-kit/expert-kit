"""Run the required small Qwen generation check through Expert Kit."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_NEW_TOKENS = 20
_DEFAULT_PROMPT = "What is a mixture-of-experts model?"


def _validate_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Qwen smoke expected exactly one generated result")
    result = results[0]
    if not isinstance(result, dict):
        raise RuntimeError("Qwen smoke returned a malformed result")

    output_tokens = result.get("output_tokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or not 1 <= output_tokens <= _MAX_NEW_TOKENS
    ):
        raise RuntimeError("Qwen smoke did not generate between 1 and 20 tokens")

    generated_text = " ".join(
        value.strip()
        for key in ("thinking_content", "content")
        if isinstance((value := result.get(key)), str) and value.strip()
    )
    if not generated_text:
        raise RuntimeError("Qwen smoke generated no readable text")
    return result


def run_qwen_smoke(
    *,
    model_path: str,
    controller_endpoint: str,
    instance_id: int,
    prompt: str = _DEFAULT_PROMPT,
) -> dict[str, Any]:
    """Generate one response with the fixed migration-gate limits.

    Args:
        model_path: Local path to a supported Qwen model.
        controller_endpoint: Plaintext Controller gRPC address in ``host:port`` form.
        instance_id: Numeric model instance registered with the Controller.
        prompt: Single input prompt. Passing multiple prompts is intentionally unsupported.

    Returns:
        The one validated result produced with batch size one and at most 20 new tokens.

    Raises:
        ValueError: An argument is empty or invalid.
        RuntimeError: Generation returns no usable text or violates the fixed limits.
    """

    if not model_path.strip():
        raise ValueError("model_path must not be empty")
    if not controller_endpoint.strip():
        raise ValueError("controller_endpoint must not be empty")
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise ValueError("instance_id must be a positive integer")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from expertkit_torch.models.qwen3_moe import evaluate_batch

    payload = evaluate_batch(
        model_path=model_path,
        prompts=[prompt],
        output_max_length=_MAX_NEW_TOKENS,
        enable_ek=True,
        ek_addr=controller_endpoint,
        ek_instance_id=instance_id,
    )
    return _validate_result(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ek-qwen-smoke")
    parser.add_argument("--model-path", required=True, help="Path to the Qwen model directory")
    parser.add_argument(
        "--controller-endpoint",
        default="127.0.0.1:5002",
        help="Plaintext Controller gRPC address",
    )
    parser.add_argument("--instance-id", type=int, default=1)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fixed-size Qwen generation and print the generated text."""

    arguments = _parser().parse_args(argv)
    result = run_qwen_smoke(
        model_path=arguments.model_path,
        controller_endpoint=arguments.controller_endpoint,
        instance_id=arguments.instance_id,
        prompt=arguments.prompt,
    )
    generated_text = " ".join(
        value.strip()
        for key in ("thinking_content", "content")
        if isinstance((value := result.get(key)), str) and value.strip()
    )
    print(f"Qwen smoke passed: {result['output_tokens']} generated tokens")
    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
