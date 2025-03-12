import re
from typing import Dict


def create_splitting_plan(weight_map: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Create a splitting plan based on weight names.

    Args:
        weight_map: Original weight map from safetensors index file

    Returns:
        Dict mapping new file names to their contents (which are dicts of weight_name -> original_file)
    """
    new_mapping = {}

    # Iterate through all weights
    for weight_name, original_file in weight_map.items():
        # Extract layer index (if present)
        layer_match = re.search(r"model\.layers\.(\d+)", weight_name)
        layer_idx = int(layer_match.group(1)) if layer_match else None

        # Determine which file this weight should go into
        if layer_idx is not None and layer_idx < 3:
            # Hard code for DeepSeek-R1. As the first 3 layers of R1 is dense layer, merge in model-share ckpt.
            new_file = "model-share.safetensors"
        elif "mlp.experts" in weight_name:
            # Extract expert number
            expert_match = re.search(r"mlp\.experts\.(\d+)", weight_name)
            if layer_idx is not None and expert_match:
                expert_idx = expert_match.group(1)
                new_file = f"model-layer{layer_idx}-expert{expert_idx}.safetensors"
            else:
                # If unable to extract, default to shared file
                new_file = "model-share.safetensors"
        elif "mlp.shared_experts" in weight_name or "mlp.gate" in weight_name:
            if layer_idx is not None:
                new_file = f"model-layer{layer_idx}-shared.safetensors"
            else:
                new_file = "model-share.safetensors"
        else:
            # All non-expert weights (attention, layernorms, tokens, and regular MLPs)
            # go into the shared file
            new_file = "model-share.safetensors"

        # Add this weight to the new mapping
        if new_file not in new_mapping:
            new_mapping[new_file] = {}
        new_mapping[new_file][weight_name] = original_file

    return new_mapping
