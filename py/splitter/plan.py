import argparse
import json
import logging
import os
from tqdm import tqdm
import re
from typing import Dict

from utils.storage import DALStorage

logger = logging.getLogger("planner")

DEFAULT_SPLIT_PLAN_NAME = "splitting_plan.json"


def with_split_plan_name_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--split_plan",
        type=str,
        default=DEFAULT_SPLIT_PLAN_NAME,
    )


class SplitPlan:

    def __init__(self, *, plan: Dict[str, Dict[str, str]]):
        if not isinstance(plan, dict):
            raise ValueError("plan in wrong format")
        self.plan = plan
        return

    def expert_files(self):
        files = self.output_files()
        return [f for f in files if "expert" in f]

    def shared_file(self):
        files = self.output_files()
        return [f for f in files if "share" in f]

    def output_files(self):
        return self.plan.keys()

    def output_count(self):
        return len(self.plan.items())

    def get_source_file(self, output_file: str):
        src = self.plan[output_file]
        if src is None:
            raise ValueError(f"Source file not found for {output_file}")
        return src

    def to_plain_weight_map(self):
        """
        Convert the splitting plan to a plain weight map which has format
         tensor_name -> file_name
        """
        wm: Dict[str, str] = {}
        for new_file, weights in self.plan.items():
            for weight_name in weights.keys():
                wm[weight_name] = new_file
        return wm

    @classmethod
    def load(cls, storage: DALStorage, name=DEFAULT_SPLIT_PLAN_NAME):
        """Load the splitting plan from a JSON file"""
        obj = storage.load_json(name)
        if obj is None:
            raise FileNotFoundError(f"Splitting plan not found: {name}")
        return cls(plan=obj)

    @classmethod
    def load_local(cls, filename: str):
        """Load the splitting plan from a JSON file"""
        with open(filename, "r") as f:
            plan = json.load(f)
            return cls(plan=plan)

    async def save(self, *, local_fs: str | None, storage: DALStorage | None):
        """Save the splitting plan to local file or remote storage"""
        if local_fs:
            await self._save_local(local_fs)
        if storage:
            await self._save_to_storage(storage)

    async def _save_local(self, root: str, name=DEFAULT_SPLIT_PLAN_NAME):
        plan_file_local = os.path.join(root, name)
        with open(plan_file_local, "w") as f:
            json.dump(self.plan, f, indent=2)
        logger.info(f"Saved splitting plan to local file: {plan_file_local}")

    async def _save_to_storage(self, storage: DALStorage, name=DEFAULT_SPLIT_PLAN_NAME):
        """Save the splitting plan to remote storage"""
        # Save the plan to output storage
        await storage.save_json(name, self.plan)
        logger.info("Saved splitting plan to storage")

    @classmethod
    def create_from_index(cls, index_file_path: str):
        """Create a splitting plan from an index file"""
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"Model index file not found: {index_file_path}")

        with open(index_file_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        splitting_plan = create_splitting_plan(weight_map)
        logger.info(
            f"Generated new splitting plan with {splitting_plan.output_count()} output files"
        )
        return splitting_plan


async def check_remote_files(args, plan: SplitPlan, storage: DALStorage):
    """Check if all files defined in splitting_plan exist in remote storage"""
    logger.info(f"Checking if {plan.output_count()} files exist in remote storage...")

    # Get all planned files
    planned_files = set(plan.output_files())

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
        with tqdm(total=plan.output_count(), desc="Checking remote files") as pbar:
            for file_name in plan.output_files():
                exists = await storage.file_exists_async(file_name)
                if not exists:
                    missing_files.append(file_name)
                pbar.update(1)

    # Output results
    if missing_files:
        logger.info(
            f"Found {len(missing_files)} missing files out of {plan.output_files()} total files"
        )

        # Save missing files to output file
        with open(args.missing_files_output, "w") as f:
            for file_name in missing_files:
                f.write(f"{file_name}\n")

        logger.info(f"Missing files list saved to {args.missing_files_output}")
    else:
        logger.info("All files exist in remote storage! ✓")

    return missing_files


def create_splitting_plan(weight_map: Dict[str, str]) -> SplitPlan:
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
        # default to model-share.safetensors
        new_file = "model-share.safetensors"

        if layer_match is None:
            layer_match = re.search(r"^layers\.(\d+)", weight_name)
            layer_idx = int(layer_match.group(1)) if layer_match else None
        # Determine which file this weight should go into
        # TODO: hard code for DeepSeek-R1. As the first 3 layers of R1 is dense layer, merge in model-share ckpt.
        if layer_idx is not None and layer_idx < 3:
            new_file = "model-share.safetensors"
        elif "ffn.experts" in weight_name:
            expert_match = re.search(r"ffn\.experts\.(\d+)", weight_name)
            if layer_idx is not None and expert_match:
                expert_idx = expert_match.group(1)
                new_file = f"model-layer{layer_idx}-expert{expert_idx}.safetensors"
        elif "mlp.experts" in weight_name:
            # Extract expert number
            expert_match = re.search(r"mlp\.experts\.(\d+)", weight_name)
            if layer_idx is not None and expert_match:
                expert_idx = expert_match.group(1)
                new_file = f"model-layer{layer_idx}-expert{expert_idx}.safetensors"
        elif "mlp.shared_experts" in weight_name or "mlp.gate" in weight_name:
            if layer_idx is not None:
                new_file = f"model-layer{layer_idx}-shared.safetensors"
        elif "ffn.shared_experts" in weight_name:
            if layer_idx is not None:
                new_file = f"model-layer{layer_idx}-shared.safetensors"
        else:
            pass

        # Add this weight to the new mapping
        if new_file not in new_mapping:
            new_mapping[new_file] = {}
        new_mapping[new_file][weight_name] = original_file

    return SplitPlan(plan=new_mapping)
