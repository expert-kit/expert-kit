"""vLLM general plugin registration for remote Routed-MoE execution."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def register() -> None:
    """Replace the vLLM 0.25.1 FusedMoE factory when explicitly enabled."""

    if os.getenv("EK_ENABLE") != "1":
        return

    import vllm.model_executor.layers.fused_moe as fused_moe_package
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    from expertkit_vllm.experts.remote_moe import remote_fused_moe

    fused_moe_layer.FusedMoE = remote_fused_moe
    fused_moe_package.FusedMoE = remote_fused_moe
    logger.info("enabled Expert Kit remote MoE for vLLM 0.25.1")
