#!/usr/bin/env python3
import traceback
import time
import argparse
import asyncio
import uvloop
import os
from typing import List, Optional
from utils.storage import setup_storage, with_storage_args
from utils.base import getLogger
from splitter.plan import SplitPlan, check_remote_files
from utils.storage import DALStorage
import logging
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save

MAX_RETRY_CNT = 3
logger = getLogger(__name__)


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
        "--model_idx_file",
        type=str,
        required=False,
        default="",
        help="name of the model index file (default: xxx.index.json under model_dir)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=".",
        help="Directory for storing state files and local copies of the plan (default: current directory)",
    )

    # Storage options
    with_storage_args(parser)

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

    # Option for checking remote files
    parser.add_argument(
        "--check_remote",
        action="store_true",
        help="Check if all files in splitting_plan.json exist in remote storage",
    )
    parser.add_argument(
        "--upload_missing",
        action="store_true",
        help="Check for missing files in remote storage and re-upload them",
    )
    parser.add_argument(
        "--missing_files_output",
        type=str,
        default="missing_files.txt",
        help="File to output missing files list (default: missing_files.txt)",
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


opened_st = {}


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
        opened_st[file_path] = f

    for name in weight_names:
        if name in f.keys():
            tensors[name] = f.get_tensor(name)
    return tensors


def process_file_sync(args, output_file, weights, storage: DALStorage):
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
    st_bytes = save(all_tensors)
    return st_bytes


async def process_file(
    args,
    output_file,
    weights,
    storage: DALStorage,
):
    start = time.perf_counter()
    try:
        future = asyncio.gather(
            asyncio.to_thread(process_file_sync, args, output_file, weights, storage)
        )
        st_bytes = (await asyncio.wait_for(future, 300))[0]
        if not args.skip_upload:
            success = await storage.save_file_async(output_file, st_bytes)
            if success:
                pass
                # Mark as completed
            else:
                raise ValueError(f"Failed to save {output_file}")
        else:
            logger.info(f"[SKIP] Would save {output_file}")

        return True

    except Exception as e:
        print(traceback.format_exc())

        logger.error(f"Error processing {output_file}: {e}")
        return False

    finally:
        now = time.perf_counter()


async def process_split(
    args,
    splitting_plan: SplitPlan,
    storage: DALStorage,
    files_to_process: Optional[list] = None,
):
    """Process the actual splitting according to the plan"""
    completed_files = []

    if files_to_process is None:
        files_to_process = []
        for file_name in splitting_plan.output_files():
            if file_name not in completed_files:
                files_to_process.append(file_name)

    logger.info(
        f"Processing {len(files_to_process)} out of {splitting_plan.output_count()} files"
    )

    parallelism = 16
    task_q = asyncio.Queue(parallelism)
    done_q = asyncio.Queue(parallelism)

    async def display_progress(dq: asyncio.Queue):
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            while True:
                try:
                    await dq.get()
                except asyncio.CancelledError:
                    return
                pbar.update(1)
                dq.task_done()
                await asyncio.sleep(0)

    async def worker(q: asyncio.Queue, dq: asyncio.Queue):
        while True:
            claimed = False
            try_cnt = 0
            try:
                output_file = await q.get()
                claimed = True
                weights = splitting_plan.get_source_file(output_file)
                while try_cnt < MAX_RETRY_CNT:
                    fut = process_file(args, output_file, weights, storage)
                    success = await asyncio.wait_for(fut, 600)
                    if success:
                        break
                    try_cnt += 1
                    logger.warning(f"retry {try_cnt} for {output_file}")
                await dq.put(1)
            except Exception:
                print(traceback.format_exc())
            finally:
                if claimed:
                    q.task_done()

    async def feeder(q: asyncio.Queue):
        logger.info(
            f"Feeding files to the queue {files_to_process=}",
        )
        for output_file in files_to_process:
            if q.full():
                await asyncio.sleep(1)
                continue
            await q.put(output_file)
            await asyncio.sleep(0)

    workers = [asyncio.create_task(worker(task_q, done_q)) for _ in range(parallelism)]
    producer = asyncio.create_task(feeder(task_q))
    display = asyncio.create_task(display_progress(done_q))

    await asyncio.gather(producer)
    await task_q.join()
    await done_q.join()
    for w in workers:
        w.cancel()
    display.cancel()


# Function to handle clean exit
def handle_exit(storage=None, args=None, splitting_plan: SplitPlan | None = None):
    """Handle program exit by saving state and plan"""
    logger.info("Program exiting, performing cleanup tasks...")


async def read_or_create_plan(args, storage):
    model_index_path = os.path.join(args.model_dir, args.model_idx_file)
    plan = None
    # Load or generate splitting plan
    if args.plan_file:
        plan = SplitPlan.load_local(args.plan_file)
    else:
        logger.info("Creating new splitting plan from index file")
        plan = SplitPlan.create_from_index(model_index_path)
        await plan.save(local_fs=args.work_dir, storage=storage)

    return plan


async def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Set up storage
    storage = setup_storage(args)

    # Initialize status dict
    completed = []
    failed = []

    splitting_plan = await read_or_create_plan(args, storage=storage)

    # If check_remote option is specified, check remote files
    if args.check_remote or args.upload_missing:
        missing_files = await check_remote_files(args, splitting_plan, storage)
        if not missing_files:
            logger.info("All files from splitting plan exist in remote storage.")
        else:
            logger.info(f"{len(missing_files)} files are missing from remote storage.")
            logger.info(f"Missing files list saved to {args.missing_files_output}")

            # If upload_missing flag is set, process the missing files
            if args.upload_missing:
                logger.info(f"Starting to upload {len(missing_files)} missing files...")
                await process_split(args, splitting_plan, storage, missing_files)
                logger.info("Re-upload of missing files completed!")

        if not args.upload_missing:
            return

    # If plan_only, exit here
    if args.plan_only and not args.upload_missing:
        logger.info("Plan generated. Exiting as requested.")
        return

    # Process the splitting for all files if not just uploading missing ones
    if not args.upload_missing:
        await process_split(args, splitting_plan, storage)

    # Create a new index file

    new_index = {"weight_map": splitting_plan.to_plain_weight_map()}
    await storage.save_json(args.model_idx_file, new_index)

    # Calculate stats
    logger.info(f"Total files: {splitting_plan.output_count()}")
    logger.info(f"Successfully processed: {len(completed)}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.warning(
            "Some files failed to process. See file_status.json for details."
        )


if __name__ == "__main__":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    uvloop.run(main())
