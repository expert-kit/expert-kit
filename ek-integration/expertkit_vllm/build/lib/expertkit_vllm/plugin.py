import os
# from vllm import ModelRegistry
import vllm.model_executor.models.deepseek_v2 as ds_v2
from .expertkit_moe import ExpertKitMoE


def register():
    """Register the ExpertKit plugin with vLLM.

    This function is called by vLLM's plugin system during initialization.
    It replaces the DeepseekV2MoE implementation with the ExpertKitMoE
    implementation when the EXPERTKIT_ENABLE environment variable is set.
    """
    # Only activate plugin when explicitly enabled
    if os.getenv("EXPERTKIT_ENABLE") != "1":
        return

    # Replace DeepseekV2MoE with ExpertKitMoE
    ds_v2.DeepseekV2MoE = ExpertKitMoE

    #TODO: change model loading logic