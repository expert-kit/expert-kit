#!/usr/bin/env python3
import traceback
import time
import argparse
import asyncio
import concurrent
import uvloop
import json
import os
import sys
import time
import io
import atexit
import signal
from typing import Dict, List
from splitter.tracker import FileProcessTracker
from splitter.planner import create_splitting_plan
from splitter.dal import DALStorage
import logging
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("model_splitter")




def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Split DeepSeek-R1 model weights")

    # Input/Output locations
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the original model files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving the split model files or plan",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Directory for storing state files and local copies of the plan (default: current directory)",
    )

    # Storage options
    parser.add_argument(
        "--storage_type",
        type=str,
        choices=["fs", "s3", "oss"],
        default="fs",
        help="Storage type: filesystem (fs) or S3 bucket (s3) (default: fs)",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        help="S3 bucket name (required if storage_type is {s3, oss})",
    )
    parser.add_argument(
        "--s3_prefix", type=str, default="", help='Prefix for S3 objects (default: "")'
    )
    parser.add_argument(
        "--access_key", type=str, help="AWS Access Key ID for S3 access"
    )
    parser.add_argument(
        "--access_secret", type=str, help="AWS Secret Access Key for S3 access"
    )
    parser.add_argument(
        "--s3_endpoint_url",
        type=str,
        help="Custom S3 endpoint URL for S3-compatible services (e.g., Aliyun OSS, MinIO)",
    )
    parser.add_argument(
        "--s3_region",
        type=str,
        default="us-east-1",
        help="AWS region for S3 (default: us-east-1)",
    )

    # Operation mode
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Only generate the splitting plan without creating the files",
    )
    parser.add_argument(
        "--plan_file",
        type=str,
        help="Use an existing splitting plan file instead of generating a new one",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted splitting operation",
    )
    parser.add_argument(
        "--skip_upload",
        action="store_true",
        help="Do not upload/save the split files (useful for testing)",
    )

    # Performance options
    parser.add_argument(
        "--thread_num",
        type=int,
        default=1,
        help="Number of files to process in parallel (default: 1)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.storage_type in ["s3", "oss"] and not args.s3_bucket:
        parser.error("--s3_bucket is required when --storage_type is {s3, oss}")

    if args.plan_file and not os.path.isfile(args.plan_file):
        parser.error(f"Plan file not found: {args.plan_file}")

    # Ensure work_dir exists
    os.makedirs(args.work_dir, exist_ok=True)

    return args


def setup_storage(args):
    """Set up storage handler based on arguments"""
    if args.storage_type in ["s3", "oss"]:
        # Create S3 storage
        return DALStorage(
            args.storage_type,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            access_key=args.access_key,
            secret_key=args.access_secret,
            endpoint=args.s3_endpoint_url,
            region=args.s3_region,
        )
    else:
        # Create filesystem storage
        return DALStorage("fs", root=args.output_dir)


opened_st= {}

def load_safetensor_file(file_path: str, weight_names: List[str]):
    """
    Load specific weights from a safetensor file

    Args:
        file_path: Path to the safetensor file
        weight_names: List of weight names to extract

    Returns:
        Dictionary mapping weight names to their tensor data
    """
    tensors = {}
    f = opened_st.get(file_path)
    if f is None:
        f = safe_open(file_path, framework="pt") 
        opened_st[file_path] =f

    for name in weight_names:
        if name in f.keys():
            tensors[name] = f.get_tensor(name)
    return tensors



def process_file_sync(
    args,
    output_file,
    weights,
    tracker: FileProcessTracker,
    storage: DALStorage):
    tracker.mark_in_progress(output_file)
    logger.debug(f"Processing {output_file} with {len(weights)} weights")
    # Group weights by source file for efficient loading
    by_source = {}
    for weight_name, source_file in weights.items():
        if source_file not in by_source:
            by_source[source_file] = []
        by_source[source_file].append(weight_name)

    # Extract weights from each source file
    all_tensors = {}
    for source_file, weight_names in by_source.items():
        source_path = os.path.join(args.model_dir, source_file)
        tensors = load_safetensor_file(source_path, weight_names)
        all_tensors.update(tensors)

    if not all_tensors:
        raise ValueError(f"No weights found for {output_file}")

    # Create the new safetensor file
    st_bytes= save(all_tensors)
    return st_bytes

async def process_file(
    args,
    output_file,
    weights,
    tracker: FileProcessTracker,
    storage: DALStorage,
):
    logger.info(f"process {output_file} start ")
    start = time.perf_counter()
    try:
        future = asyncio.gather(asyncio.to_thread(process_file_sync,args,output_file,weights,tracker,storage))
        st_bytes = await asyncio.wait_for(future, 30)
        if not args.skip_upload:
            success = await storage.save_file_async(output_file, st_bytes)
            if success:
                # Mark as completed
                tracker.mark_completed(output_file)
            else:
                raise ValueError(f"Failed to save {output_file}")
        else:
            logger.info(f"[SKIP] Would save {output_file}")
           # tracker.mark_completed(output_file)

        return True

    except Exception as e:
        print(traceback.format_exc())

        logger.error(f"Error processing {output_file}: {e}")
        tracker.mark_failed(output_file, str(e))
        return False

    finally:
        now = time.perf_counter()
        logger.info(f"process {output_file} done, elapsed = {now-start}")

    


async def process_split(args, splitting_plan, storage, tracker):
    """Process the actual splitting according to the plan"""
    completed_files = set(tracker.get_completed_files())
    files_to_process = []

    for file_name in splitting_plan.keys():
        if file_name not in completed_files:
            files_to_process.append(file_name)

    # Update the tracker with pending files
    tracker.mark_pending(files_to_process)

    logger.info(
        f"Processing {len(files_to_process)} out of {len(splitting_plan)} files"
    )

    parallelism = 16
    task_q = asyncio.Queue(parallelism)
    done_q = asyncio.Queue(parallelism)

    async def display_progress(dq: asyncio.Queue):
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            while True:
                await dq.get()
                pbar.update(1)
                dq.task_done()
                await asyncio.sleep(0)

    async def worker(q: asyncio.Queue, dq: asyncio.Queue):
        while True:
            output_file = await q.get()
            weights = splitting_plan[output_file]
            fut  =  process_file(args, output_file, weights, tracker, storage)
            await asyncio.wait_for(fut, 30)
            q.task_done()

    async def feeder(q: asyncio.Queue):
        for output_file in files_to_process:
            if q.full():
                await asyncio.sleep(1)
            await q.put(output_file)
            await asyncio.sleep(0)

    workers = [asyncio.create_task(worker(task_q,done_q)) for _ in range(parallelism)]
    producer = asyncio.create_task(feeder(task_q))

    await asyncio.gather(producer)
    await task_q.join()
    await done_q.join()
    for w in workers:
        w.cancel()
    display.cancel()

    # Check for failed files
    failed_files = tracker.get_failed_files()
    if failed_files:
        logger.warning(f"{len(failed_files)} files failed to process")
        for f, error in failed_files.items():
            logger.warning(f"  - {f}: {error['error']}")


# Function to handle clean exit
def handle_exit(tracker=None, storage=None, args=None, splitting_plan=None):
    """Handle program exit by saving state and plan"""
    logger.info("Program exiting, performing cleanup tasks...")

    if tracker is not None:
        logger.info("Backing up tracker state...")
        # First ensure the local status file is up-to-date
        tracker._save()
        # Then back it up to remote storage
        tracker.backup_to_storage()

    if splitting_plan is not None:
        # First save locally if args is available
        if args is not None:
            plan_file_local = os.path.join(args.work_dir, "splitting_plan.json")
            with open(plan_file_local, "w") as f:
                json.dump(splitting_plan, f, indent=2)
            logger.info(
                f"Saved final splitting plan to local file: {plan_file_local}"
            )

        # Then back up to storage
        if storage is not None:
            logger.info("Backing up splitting plan to storage...")
            # Save plan to storage
            storage.save_json("splitting_plan.json", splitting_plan)


async def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Set up storage
    storage = setup_storage(args)

    # Initialize file tracker with the work_dir and storage for backups
    tracker = FileProcessTracker(work_dir=args.work_dir, storage=storage)

    # Load original weight map
    model_index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if not os.path.isfile(model_index_path):
        logger.error(f"Model index file not found: {model_index_path}")
        sys.exit(1)

    with open(model_index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    # Generate or load splitting plan
    if args.plan_file:
        with open(args.plan_file, "r") as f:
            splitting_plan = json.load(f)
        logger.info(f"Loaded splitting plan from {args.plan_file}")
    else:
        splitting_plan = create_splitting_plan(weight_map)
        logger.info(
            f"Generated new splitting plan with {len(splitting_plan)} output files"
        )

        # Save the plan to both work_dir and storage
        # 1. Save to work_dir
        plan_file_local = os.path.join(args.work_dir, "splitting_plan.json")
        with open(plan_file_local, "w") as f:
            json.dump(splitting_plan, f, indent=2)
        logger.info(f"Saved splitting plan to local file: {plan_file_local}")

        # 2. Save to output storage
        if not args.skip_upload:
            await storage.save_json("splitting_plan.json", splitting_plan)
            logger.info("Saved splitting plan to storage")

    # Register exit handler to backup state on program exit
    atexit.register(
        handle_exit,
        tracker=tracker,
        storage=storage,
        args=args,
        splitting_plan=splitting_plan,
    )

    # Register signal handlers for common termination signals
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(
            sig,
            lambda s, f: (
                logger.warning(f"Received signal {s}, shutting down..."),
                #handle_exit(tracker, storage, args, splitting_plan),
                sys.exit(1),
            ),
        )

    # If plan_only, exit here
    if args.plan_only:
        logger.info("Plan generated. Exiting as requested.")
        return

    # Process the splitting
    await process_split(args, splitting_plan, storage, tracker)

    # Create a new index file
    new_index = {"weight_map": {}}
    for new_file, weights in splitting_plan.items():
        for weight_name in weights.keys():
            new_index["weight_map"][weight_name] = new_file

    await storage.save_json("model.safetensors.index.json", new_index)

    # Calculate stats
    completed = tracker.get_completed_files()
    failed = tracker.get_failed_files()
    status = tracker.status
    duration = time.time() - status["start_time"]

    logger.info(f"Splitting completed in {duration:.2f} seconds!")
    logger.info(f"Total files: {len(splitting_plan)}")
    logger.info(f"Successfully processed: {len(completed)}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.warning(
            "Some files failed to process. See file_status.json for details."
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    uvloop.run(main())
