import logging
from tqdm import tqdm
import re
from typing import Dict

from .dal import DALStorage

logger = logging.getLogger("planner")


async def check_remote_files(args, splitting_plan, storage: DALStorage):
    """Check if all files defined in splitting_plan exist in remote storage"""
    logger.info(
        f"Checking if {len(splitting_plan)} files exist in remote storage...")

    # Get all planned files
    planned_files = set(splitting_plan.keys())

    # Get all existing files in one operation
    try:
        logger.info("Fetching remote file list...")
        all_files = storage.list_files("")
        existing_files = set(all_files)

        logger.info(f"Found {len(existing_files)} files in remote storage")

        # Find missing files by set difference
        missing_files = list(planned_files - existing_files)

    except Exception as e:
        # Fallback to one-by-one checking if bulk listing fails
        logger.info(
            f"Bulk listing failed ({str(e)}), falling back to individual file checks"
        )
        missing_files = []

        # Create a progress bar for checking files
        with tqdm(total=len(splitting_plan), desc="Checking remote files") as pbar:
            for file_name in splitting_plan.keys():
                exists = await storage.file_exists_async(file_name)
                if not exists:
                    missing_files.append(file_name)
                pbar.update(1)

    # Output results
    if missing_files:
        logger.info(
            f"Found {len(missing_files)} missing files out of {len(splitting_plan)} total files"
        )

        # Save missing files to output file
        with open(args.missing_files_output, "w") as f:
            for file_name in missing_files:
                f.write(f"{file_name}\n")

        logger.info(f"Missing files list saved to {args.missing_files_output}")
    else:
        logger.info("All files exist in remote storage! ✓")

    return missing_files


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
