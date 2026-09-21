"""vLLM general plugin registration for remote Routed-MoE execution."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register() -> None:
    """Wrap the active vLLM 0.25.1 FusedMoE factory when explicitly enabled."""

    if os.getenv("EK_ENABLE") != "1":
        return

    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        # General-plugin ordering is not a platform contract. Force the public
        # Ascend pre-registration hook to install its FusedMoE wrapper before
        # Expert Kit captures the active factory.
        current_platform.pre_register_and_update()

    import vllm.model_executor.layers.fused_moe as fused_moe_package
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    from expertkit_vllm.experts.remote_moe import (
        is_expertkit_fused_moe_factory,
        wrap_fused_moe_factory,
    )

    if is_expertkit_fused_moe_factory(fused_moe_layer.FusedMoE):
        return

    remote_factory = wrap_fused_moe_factory(fused_moe_layer.FusedMoE)
    # These are deliberately mutable module exports: model modules import the
    # package binding, while the platform wrapper owns the layer binding.
    fused_moe_layer.__dict__["FusedMoE"] = remote_factory
    fused_moe_package.__dict__["FusedMoE"] = remote_factory
    logger.info(f"enabled Expert Kit remote MoE for vLLM 0.25.1 on {current_platform.device_type}")
