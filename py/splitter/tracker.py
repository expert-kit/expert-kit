import os, json, time, threading, logging

logger = logging.getLogger("tracker")


class FileProcessTracker:
    """Tracks the status of files being processed"""

    def __init__(
        self,
        work_dir=".",
        storage=None,
        status_filename="file_status.json",
        backup_interval=60,
    ):
        self.work_dir = work_dir
        self.filename = status_filename
        self.backup_filename = status_filename.replace(".json", ".bak.json")
        self.local_path = os.path.join(self.work_dir, self.filename)
        self.local_backup_path = os.path.join(self.work_dir, self.backup_filename)
        self.storage = storage  # Storage handler for remote backup
        self.lock = threading.Lock()
        self.backup_interval = backup_interval  # Backup interval in seconds
        self.last_backup_time = time.time()

        # Try to load existing status from local file first
        if os.path.exists(self.local_path):
            try:
                with open(self.local_path, "r") as f:
                    self.status = json.load(f)
                logger.info(f"Loaded file status from local file: {self.local_path}")
            except Exception as e:
                logger.warning(f"Failed to load local status file: {e}")
                # try to recover from backup file
                if os.path.exists(self.local_backup_path):
                    try:
                        with open(self.local_backup_path, "r") as f:
                            self.status = json.load(f)
                        logger.info(
                            f"Recovered from backup file: {self.local_backup_path}"
                        )
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
            target=self._periodic_backup_thread, daemon=True
        )
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
                os.makedirs(os.path.dirname(self.local_backup_path), exist_ok=True)
                with open(self.local_backup_path, "w") as f:
                    json.dump(self.status, f, indent=2)

                logger.debug(f"Created local backup at {self.local_backup_path}")

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
            "pending": [],  # Files waiting to be processed
            "in_progress": {},  # Files currently being processed with start time
            "completed": [],  # Successfully processed files
            "failed": {},  # Failed files with error messages
            "start_time": time.time(),
            "last_updated": time.time(),
        }

        # Ensure the status is immediately saved to disk
        try:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            with open(self.local_path, "w") as f:
                json.dump(status, f, indent=2)
            logger.info(
                f"Created new status file at: {self.local_path}, with status: {status}"
            )
        except Exception as e:
            logger.error(f"Failed to create initial status file: {e}")

        return status

    def mark_pending(self, files):
        """Mark files as pending processing"""
        with self.lock:
            self.status["pending"] = list(set(files) - set(self.status["completed"]))
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
            self.status["failed"][filename] = {"error": str(error), "time": time.time()}
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
            with open(self.local_path, "w") as f:
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
                        f"Local status file doesn't exist, creating it first"
                    )
                    with open(self.local_path, "w") as f:
                        json.dump(self.status, f, indent=2)

                # Now read and upload it
                with open(self.local_path, "r") as f:
                    status_data = f.read()

                storage_success = self.storage.save_file(
                    self.filename, status_data.encode("utf-8")
                )
                if storage_success:
                    logger.info(f"Backed up status file to storage at {self.filename}")
                else:
                    logger.warning(f"Failed to backup status file to storage")

        except Exception as e:
            logger.error(f"Error backing up status file to storage: {e}")
