import logging
import sys
import json
from typing import Optional, List

logger = logging.getLogger("dal")
try:
    import opendal
except ImportError:
    logger.error("OpenDAL is required. Install with: pip install opendal")
    sys.exit(1)


class DALStorage:
    """Storage handler using OpenDAL"""

    def __init__(self, storage_type: str, **kwargs):
        """Initialize storage with OpenDAL Operator"""
        self.storage_type = storage_type

        # Create OpenDAL operator based on storage type
        if storage_type == "fs":
            self.operator = opendal.Operator("fs", root=kwargs.get("root", "./"))
            self.async_operator = opendal.AsyncOperator("fs", root=kwargs.get("root", "./"))
        elif storage_type in ["s3", "oss"]:
            # Handle S3-compatible services
            s3_config = {
                "root": kwargs.get("prefix", ""),
                "bucket": kwargs.get("bucket"),
                "region": kwargs.get("region", "us-east-1"),
            }

            access_key = kwargs.get("access_key")
            secret_key = kwargs.get("secret_key")

            assert access_key is not None
            assert secret_key is not None
            match storage_type:
                case "s3":
                    s3_config["access_key"] = access_key
                    s3_config["secret_key"] = secret_key
                case "oss":
                    s3_config["access_key_id"] = access_key
                    s3_config["access_key_secret"] = secret_key

            # Add custom endpoint for non-AWS S3 services (like Aliyun OSS, MinIO, etc.)
            if kwargs.get("endpoint"):
                s3_config["endpoint"] = kwargs.get("endpoint")

            self.operator = opendal.Operator(storage_type, **s3_config)
            self.async_operator = opendal.AsyncOperator(storage_type, **s3_config)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        self._file_exists_cache = set()

    async def save_file_async(self, filename: str, data: bytes) -> bool:
        """Save a file to storage asynchronously"""
        await self.async_operator.write(filename, data)
        self._file_exists_cache.add(filename)
        return True

    async def file_exists_async(self, filename: str) -> bool:
        """Check if a file exists in storage or cache asynchronously"""
        if filename in self._file_exists_cache:
            return True

        try:
            # Try to stat the file
            await self.async_operator.stat(filename)
            self._file_exists_cache.add(filename)
            return True
        except Exception:
            import traceback

            print(traceback.format_exc())
            return False

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in storage or cache"""
        if filename in self._file_exists_cache:
            return True

        # Try to stat the file
        self.operator.stat(filename)
        self._file_exists_cache.add(filename)
        return True

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage"""
        return [str(file) for file in self.operator.list(prefix)]

    def load_data(self, filename: str) -> Optional[bytes]:
        """Load binary data from a file"""
        return self.operator.read(filename)

    def load_json(self, filename: str) -> Optional[dict]:
        """Load JSON data from a file"""
        data = self.operator.read(filename)
        return json.loads(data.decode("utf-8"))

    async def save_json(self, filename: str, data: dict) -> bool:
        """Save JSON data to a file"""
        json_str = json.dumps(data, indent=2)
        return await self.save_file_async(filename, json_str.encode("utf-8"))
