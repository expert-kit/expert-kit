"""Tests for the Weight adapter runtime interface."""

from __future__ import annotations

import pytest

from expertkit_worker.weights import WeightAdapter


def test_weight_adapter_requires_every_conversion_and_sizing_method() -> None:
    class IncompleteAdapter(WeightAdapter[object, object]):
        pass

    with pytest.raises(TypeError, match="abstract class"):
        IncompleteAdapter()
