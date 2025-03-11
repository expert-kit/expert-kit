#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import tempfile
import threading
import atexit
import signal
import concurrent
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_splitter")

# Try to import optional dependencies
try:
    import opendal
    OPENDAL_AVAILABLE = True
except ImportError:
    OPENDAL_AVAILABLE = False

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Split DeepSeek-R1 model weights")

    # Input/Output locations
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the original model files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for saving the split model files or plan")
    parser.add_argument("--work_dir", type=str, default=".",
                        help="Directory for storing state files and local copies of the plan (default: current directory)")

    # Storage options
    parser.add_argument("--storage_type", type=str, choices=["fs", "s3", "oss"], default="fs",
                        help="Storage type: filesystem (fs) or S3 bucket (s3) (default: fs)")
    parser.add_argument("--s3_bucket", type=str,
                        help="S3 bucket name (required if storage_type is {s3, oss})")
    parser.add_argument("--s3_prefix", type=str, default="",
                        help="Prefix for S3 objects (default: \"\")")
    parser.add_argument("--access_key", type=str,
                        help="AWS Access Key ID for S3 access")
    parser.add_argument("--access_secret", type=str,
                        help="AWS Secret Access Key for S3 access")
    parser.add_argument("--s3_endpoint_url", type=str,
                        help="Custom S3 endpoint URL for S3-compatible services (e.g., Aliyun OSS, MinIO)")
    parser.add_argument("--s3_region", type=str, default="us-east-1",
                        help="AWS region for S3 (default: us-east-1)")

    # Operation mode
    parser.add_argument("--plan_only", action="store_true",
                        help="Only generate the splitting plan without creating the files")
    parser.add_argument("--plan_file", type=str,
                        help="Use an existing splitting plan file instead of generating a new one")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously interrupted splitting operation")
    parser.add_argument("--skip_upload", action="store_true",
                        help="Do not upload/save the split files (useful for testing)")

    # Performance options
    parser.add_argument("--thread_num", type=int, default=1,
                        help="Number of files to process in parallel (default: 1)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Validate arguments
    if not SAFETENSORS_AVAILABLE:
        logger.error(
            "SafeTensors is required. Install with: pip install safetensors torch")
        sys.exit(1)

    if not OPENDAL_AVAILABLE:
        logger.error(
            "OpenDAL is required. Install with: pip install opendal")
        sys.exit(1)

    if args.storage_type in ["s3", "oss"] and not args.s3_bucket:
        parser.error(
            "--s3_bucket is required when --storage_type is {s3, oss}")

    if args.plan_file and not os.path.isfile(args.plan_file):
        parser.error(f"Plan file not found: {args.plan_file}")

    # Ensure work_dir exists
    os.makedirs(args.work_dir, exist_ok=True)

    return args


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


class DALStorage:
    """Storage handler using OpenDAL"""

    def __init__(self, storage_type: str, **kwargs):
        """Initialize storage with OpenDAL Operator"""
        self.storage_type = storage_type

        # Create OpenDAL operator based on storage type
        if storage_type == "fs":
            self.operator = opendal.Operator(
                "fs", root=kwargs.get("root", "./"))
        elif storage_type in ["s3", "oss"]:
            # Handle S3-compatible services
            s3_config = {
                "root": kwargs.get("prefix", ""),
                "bucket": kwargs.get("bucket"),
                "region": kwargs.get("region", "us-east-1"),
            }

            access_key = kwargs.get("access_key")
            secret_key = kwargs.get("secret_key")

            match storage_type:
                case "s3":
                    if access_key and secret_key:
                        s3_config["access_key"] = access_key
                        s3_config["secret_key"] = secret_key
                case "oss":
                    if access_key and secret_key:
                        s3_config["access_key_id"] = access_key
                        s3_config["access_key_secret"] = secret_key

            # Add custom endpoint for non-AWS S3 services (like Aliyun OSS, MinIO, etc.)
            if kwargs.get("endpoint"):
                s3_config["endpoint"] = kwargs.get("endpoint")

            self.operator = opendal.Operator(storage_type, **s3_config)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        # Create async operator for parallel operations
        self.async_operator = opendal.AsyncOperator(
            storage_type,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

        # Cache for file existence to avoid unnecessary checks
        self._file_exists_cache = set()

    def save_file(self, filename: str, data: bytes) -> bool:
        """Save a file to storage"""
        try:
            self.operator.write(filename, data)
            self._file_exists_cache.add(filename)
            return True
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return False

    async def save_file_async(self, filename: str, data: bytes) -> bool:
        """Save a file to storage asynchronously"""
        try:
            await self.async_operator.write(filename, data)
            self._file_exists_cache.add(filename)
            return True
        except Exception as e:
            logger.error(f"Error saving file {filename}: {e}")
            return False

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in storage or cache"""
        if filename in self._file_exists_cache:
            return True

        try:
            # Try to stat the file
            self.operator.stat(filename)
            self._file_exists_cache.add(filename)
            return True
        except:
            return False

    def load_data(self, filename: str) -> Optional[bytes]:
        """Load binary data from a file"""
        try:
            return self.operator.read(filename)
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return None

    def load_json(self, filename: str) -> Optional[dict]:
        """Load JSON data from a file"""
        try:
            data = self.operator.read(filename)
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.debug(f"Could not load JSON file {filename}: {e}")
            return None

    def save_json(self, filename: str, data: dict) -> bool:
        """Save JSON data to a file"""
        json_str = json.dumps(data, indent=2)
        return self.save_file(filename, json_str.encode('utf-8'))


class FileProcessTracker:
    """Tracks the status of files being processed"""

    def __init__(self, work_dir=".", storage=None, status_filename="file_status.json", backup_interval=60):
        self.work_dir = work_dir
        self.filename = status_filename
        self.backup_filename = status_filename.replace(".json", ".bak.json")
        self.local_path = os.path.join(self.work_dir, self.filename)
        self.local_backup_path = os.path.join(
            self.work_dir, self.backup_filename)
        self.storage = storage  # Storage handler for remote backup
        self.lock = threading.Lock()
        self.backup_interval = backup_interval  # Backup interval in seconds
        self.last_backup_time = time.time()

        # Try to load existing status from local file first
        if os.path.exists(self.local_path):
            try:
                with open(self.local_path, 'r') as f:
                    self.status = json.load(f)
                logger.info(
                    f"Loaded file status from local file: {self.local_path}")
            except Exception as e:
                logger.warning(f"Failed to load local status file: {e}")
                # try to recover from backup file
                if os.path.exists(self.local_backup_path):
                    try:
                        with open(self.local_backup_path, 'r') as f:
                            self.status = json.load(f)
                        logger.info(
                            f"Recovered from backup file: {self.local_backup_path}")
                    except Exception as e2:
                        logger.warning(f"Failed to load backup file: {e2}")
                        self.status = self._create_new_status()
                else:
                    self.status = self._create_new_status()
        else:
            self.status = self._create_new_status()
            logger.info("Created new file processing status tracker")

        # start backup thread
        self.backup_thread = threading.Thread(
            target=self._periodic_backup_thread, daemon=True)
        self.backup_thread.start()

    def _periodic_backup_thread(self):
        """Periodically create backups of the status file"""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                current_time = time.time()

                # Check if it's time to create a backup
                if current_time - self.last_backup_time >= self.backup_interval:
                    logger.debug("Performing periodic backup of status file")
                    self._create_backup()
                    self.last_backup_time = current_time
            except Exception as e:
                logger.error(f"Error in backup thread: {e}")

    def _create_backup(self):
        """Create a backup of the current status file"""
        try:
            with self.lock:
                # 1. create local backup
                os.makedirs(os.path.dirname(
                    self.local_backup_path), exist_ok=True)
                with open(self.local_backup_path, 'w') as f:
                    json.dump(self.status, f, indent=2)

                logger.debug(
                    f"Created local backup at {self.local_backup_path}")

                # # 2. create remote backup if storage is available
                # if self.storage is not None:
                #     with open(self.local_backup_path, 'r') as f:
                #         backup_data = f.read()

                #     success = self.storage.save_file(
                #         self.backup_filename, backup_data.encode('utf-8'))
                #     if success:
                #         logger.debug(
                #             f"Created remote backup at {self.backup_filename}")
                #     else:
                #         logger.warning("Failed to create remote backup")
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")

    def _create_new_status(self):
        """Create a new empty status dictionary and save it to disk"""
        status = {
            "pending": [],      # Files waiting to be processed
            "in_progress": {},  # Files currently being processed with start time
            "completed": [],    # Successfully processed files
            "failed": {},       # Failed files with error messages
            "start_time": time.time(),
            "last_updated": time.time()
        }

        # Ensure the status is immediately saved to disk
        try:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            with open(self.local_path, 'w') as f:
                json.dump(status, f, indent=2)
            logger.info(
                f"Created new status file at: {self.local_path}, with status: {status}")
        except Exception as e:
            logger.error(f"Failed to create initial status file: {e}")

        return status

    def mark_pending(self, files):
        """Mark files as pending processing"""
        with self.lock:
            self.status["pending"] = list(
                set(files) - set(self.status["completed"]))
            self._save()

    def mark_in_progress(self, filename):
        """Mark a file as being processed"""
        with self.lock:
            if filename in self.status["pending"]:
                self.status["pending"].remove(filename)
            self.status["in_progress"][filename] = time.time()
            self._save()

    def mark_completed(self, filename):
        """Mark a file as successfully processed"""
        with self.lock:
            if filename in self.status["in_progress"]:
                del self.status["in_progress"][filename]
            if filename in self.status["failed"]:
                del self.status["failed"][filename]
            if filename not in self.status["completed"]:
                self.status["completed"].append(filename)
            self.status["last_updated"] = time.time()
            self._save()

    def mark_failed(self, filename, error):
        """Mark a file as failed with error message"""
        with self.lock:
            if filename in self.status["in_progress"]:
                del self.status["in_progress"][filename]
            self.status["failed"][filename] = {
                "error": str(error),
                "time": time.time()
            }
            self._save()

    def get_pending_files(self):
        """Get files that are waiting to be processed"""
        with self.lock:
            return list(self.status["pending"])

    def get_completed_files(self):
        """Get files that have been successfully processed"""
        with self.lock:
            return list(self.status["completed"])

    def get_in_progress_files(self):
        """Get files that are currently being processed"""
        with self.lock:
            return dict(self.status["in_progress"])

    def get_failed_files(self):
        """Get files that failed processing"""
        with self.lock:
            return dict(self.status["failed"])

    def is_completed(self, filename):
        """Check if a file has been successfully processed"""
        with self.lock:
            return filename in self.status["completed"]

    def _save(self):
        """Save the current status to local file and backup to storage if available"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)

            # Save to local file
            with open(self.local_path, 'w') as f:
                json.dump(self.status, f, indent=2)

            logger.debug(f"Saved status to local file: {self.local_path}")

            # Check if it's time to create a backup
            current_time = time.time()
            if current_time - self.last_backup_time >= self.backup_interval:
                self._create_backup()
                self.last_backup_time = current_time

        except Exception as e:
            logger.error(f"Failed to save status to local file: {e}")

    def backup_to_storage(self):
        """Backup the status file to remote storage if available"""
        if self.storage is None:
            return

        try:
            # Save current status to ensure it's up to date
            with self.lock:
                # Make sure the file exists in local storage first
                if not os.path.exists(self.local_path):
                    logger.warning(
                        f"Local status file doesn't exist, creating it first")
                    with open(self.local_path, 'w') as f:
                        json.dump(self.status, f, indent=2)

                # Now read and upload it
                with open(self.local_path, 'r') as f:
                    status_data = f.read()

                storage_success = self.storage.save_file(
                    self.filename, status_data.encode('utf-8'))
                if storage_success:
                    logger.info(
                        f"Backed up status file to storage at {self.filename}")
                else:
                    logger.warning(f"Failed to backup status file to storage")

        except Exception as e:
            logger.error(f"Error backing up status file to storage: {e}")


def setup_storage(args):
    """Set up storage handler based on arguments"""
    if args.storage_type in ["s3", "oss"]:
        if not OPENDAL_AVAILABLE:
            raise ImportError("OpenDAL is required for storage operations")

        # Create S3 storage
        return DALStorage(
            args.storage_type,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            access_key=args.access_key,
            secret_key=args.access_secret,
            endpoint=args.s3_endpoint_url,
            region=args.s3_region
        )
    else:
        # Create filesystem storage
        return DALStorage("fs", root=args.output_dir)


def load_safetensor_file(file_path: str, weight_names: List[str]):
    """
    Load specific weights from a safetensor file

    Args:
        file_path: Path to the safetensor file
        weight_names: List of weight names to extract

    Returns:
        Dictionary mapping weight names to their tensor data
    """
    try:
        tensors = {}
        with safe_open(file_path, framework="pt") as f:
            for name in weight_names:
                if name in f.keys():
                    tensors[name] = f.get_tensor(name)

        return tensors
    except Exception as e:
        logger.error(f"Error loading safetensor file {file_path}: {e}")
        return {}


def process_file(args, output_file, weights, temp_dir, tracker, storage):
    """
    Process a single file according to the splitting plan

    Args:
        args: Command line arguments
        output_file: Name of the output file to create
        weights: Dictionary of weights to extract
        temp_dir: Temporary directory for intermediate files
        tracker: FileProcessTracker instance
        storage: Storage handler
    """
    try:
        # Mark file as in progress
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

        # Create a temporary output file
        temp_output = os.path.join(temp_dir, output_file)
        Path(os.path.dirname(temp_output)).mkdir(parents=True, exist_ok=True)

        # Create the new safetensor file
        save_file(all_tensors, temp_output)

        # Upload to storage if not skipping
        if not args.skip_upload:
            with open(temp_output, 'rb') as f:
                data = f.read()

            success = storage.save_file(output_file, data)
            if success:
                # Mark as completed
                tracker.mark_completed(output_file)
            else:
                raise ValueError(f"Failed to save {output_file}")
        else:
            logger.info(f"[SKIP] Would save {output_file}")
            tracker.mark_completed(output_file)

        return True

    except Exception as e:
        logger.error(f"Error processing {output_file}: {e}")
        tracker.mark_failed(output_file, str(e))
        return False


def process_split(args, splitting_plan, storage, tracker):
    """Process the actual splitting according to the plan"""
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the list of files to process
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

        with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
            # with ProcessPoolExecutor(max_workers=args.thread_num) as executor:
            # Create a progress bar
            with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
                future_to_file = {}

                # Submit all tasks to the executor
                for output_file in files_to_process:
                    weights = splitting_plan[output_file]
                    future = executor.submit(
                        process_file,
                        args,
                        output_file,
                        weights,
                        temp_dir,
                        tracker,
                        storage
                    )
                    future_to_file[future] = output_file

                # Process as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    output_file = future_to_file[future]
                    try:
                        result = future.result()
                        pbar.update(1)
                        logger.debug(f"Completed processing {output_file}")
                    except Exception as e:
                        logger.error(
                            f"Processing error for {output_file}: {e}")

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
            try:
                plan_file_local = os.path.join(
                    args.work_dir, "splitting_plan.json")
                with open(plan_file_local, 'w') as f:
                    json.dump(splitting_plan, f, indent=2)
                logger.info(
                    f"Saved final splitting plan to local file: {plan_file_local}")
            except Exception as e:
                logger.error(f"Failed to save local splitting plan: {e}")

        # Then back up to storage
        if storage is not None:
            logger.info("Backing up splitting plan to storage...")
            # Save plan to storage
            storage.save_json("splitting_plan.json", splitting_plan)


def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Set up storage
    storage = setup_storage(args)

    # Initialize file tracker with the work_dir and storage for backups
    tracker = FileProcessTracker(work_dir=args.work_dir, storage=storage)

    # Load original weight map
    model_index_path = os.path.join(
        args.model_dir, "model.safetensors.index.json")
    if not os.path.isfile(model_index_path):
        logger.error(f"Model index file not found: {model_index_path}")
        sys.exit(1)

    with open(model_index_path, 'r') as f:
        weight_map = json.load(f)["weight_map"]

    # Generate or load splitting plan
    if args.plan_file:
        with open(args.plan_file, 'r') as f:
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
        with open(plan_file_local, 'w') as f:
            json.dump(splitting_plan, f, indent=2)
        logger.info(f"Saved splitting plan to local file: {plan_file_local}")

        # 2. Save to output storage
        if not args.skip_upload:
            storage.save_json("splitting_plan.json", splitting_plan)
            logger.info("Saved splitting plan to storage")

    # Register exit handler to backup state on program exit
    atexit.register(handle_exit, tracker=tracker, storage=storage,
                    args=args, splitting_plan=splitting_plan)

    # Register signal handlers for common termination signals
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: (logger.warning(f"Received signal {s}, shutting down..."),
                                         handle_exit(
                                             tracker, storage, args, splitting_plan),
                                         sys.exit(1)))

    # If plan_only, exit here
    if args.plan_only:
        logger.info("Plan generated. Exiting as requested.")
        return

    # Process the splitting
    process_split(args, splitting_plan, storage, tracker)

    # Create a new index file
    new_index = {"weight_map": {}}
    for new_file, weights in splitting_plan.items():
        for weight_name in weights.keys():
            new_index["weight_map"][weight_name] = new_file

    storage.save_json("model.safetensors.index.json", new_index)

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
            "Some files failed to process. See file_status.json for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
