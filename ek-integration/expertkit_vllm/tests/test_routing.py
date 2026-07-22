"""Verify the vLLM path uses common validation before creating a client."""

from __future__ import annotations

import ast
from pathlib import Path


def test_remote_moe_validates_router_output_before_getting_client() -> None:
    source = (Path(__file__).parents[1] / "expertkit_vllm" / "experts" / "remote_moe.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RemoteMoERunner"
    )
    forward = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_impl"
    )
    calls = [
        node.func.id
        for node in ast.walk(forward)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert calls.index("validate_and_convert_routing") < calls.index("_client_for")
