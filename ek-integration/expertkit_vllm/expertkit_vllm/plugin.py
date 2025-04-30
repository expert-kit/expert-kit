import logging
import os
# from vllm import ModelRegistry
import vllm.model_executor.models.deepseek_v2 as ds_v2
import vllm.model_executor.layers.fused_moe.layer as fused_moe
from expertkit_vllm.models.deepseek_v2 import ExpertKitMoE
from expertkit_vllm.experts.grpc_expert import GrpcExpert

logger = logging.getLogger(__name__)


def register():
    """Register the ExpertKit plugin with vLLM.

    This function is called by vLLM's plugin system during initialization.
    It replaces the DeepseekV2MoE implementation with the ExpertKitMoE
    implementation when the EXPERTKIT_ENABLE environment variable is set.
    """
    # Only activate plugin when explicitly enabled
    if os.getenv("EXPERTKIT_ENABLE") != "1":
        return
    print("🚀expertkit-vllm integration activated")
    
    mode = os.getenv("EXPERTKIT_MODE", "expert_mode")
    match mode:
        case "expert_mode":
            expert_mode_register()
        case "moe_mode":
            moe_mode_register()
        case _:
            raise ValueError(f"🚀expertkit-vllm get unknown mode: {mode}")

def expert_mode_register():
    #TODO: need test, cause A10 has limited GPU memory, too small for testing
    print("🚀expertkit-vllm integration in expert_mode mode")
    fused_moe.FusedMoE = GrpcExpert

def moe_mode_register():
    print("🚀expertkit-vllm integration in moe_mode mode")
    # Replace FusedMoE with ExpertKitFusedMoE
    #TODO: hardcode for Deepseek
    ds_v2.DeepseekV2MoE = ExpertKitMoE

    #TODO: change model loading logic