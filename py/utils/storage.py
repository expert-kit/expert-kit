import argparse
from utils.base import getLogger
import json
import opendal
from typing import Dict, Optional, List

logger = getLogger(__name__)


def with_storage_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--storage_type",
        type=str,
        choices=["fs", "s3", "oss"],
        default="fs",
        help="Storage type: filesystem (fs) or S3 bucket (s3) (default: fs)",
    )

    parser.add_argument(
        "--fs_path",
        type=str,
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


class DALStorage:
    """Storage handler using OpenDAL"""

    def __init__(self, storage_type: str, **kwargs):
        """Initialize storage with OpenDAL Operator"""
        self.storage_type = storage_type

        # Create OpenDAL operator based on storage type
        if storage_type == "fs":
            self.operator = opendal.Operator("fs", root=kwargs.get("root", "./"))
            self._config = {
                "root": kwargs.get("root", "./"),
            }
            self.async_operator = opendal.AsyncOperator("fs", root=self._config["root"])
        elif storage_type in ["s3", "oss"]:
            # Handle S3-compatible services
            self._config = {
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
                    self._config["access_key"] = access_key
                    self._config["secret_key"] = secret_key
                case "oss":
                    self._config["access_key_id"] = access_key
                    self._config["access_key_secret"] = secret_key

            # Add custom endpoint for non-AWS S3 services (like Aliyun OSS, MinIO, etc.)
            if kwargs.get("endpoint"):
                self._config["endpoint"] = kwargs.get("endpoint")

            self.operator = opendal.Operator(storage_type, **self._config)
            self.async_operator = opendal.AsyncOperator(storage_type, **self.config)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        self._file_exists_cache = set()

    def type(self) -> str:
        return self.storage_type

    def config(self) -> Dict[str, str]:
        return self._config

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


def setup_storage(args) -> DALStorage:
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
        return DALStorage("fs", root=args.fs_path)
