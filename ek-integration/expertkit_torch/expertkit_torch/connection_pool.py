"""
ConnectionPool provides LRU-based connection management for direct worker communication.

This implements a bounded connection pool with LRU eviction to prevent connection explosion
when frontends connect to many workers.
"""

import asyncio
import logging
from collections import OrderedDict
from typing import Optional

import grpc

logger = logging.getLogger(__name__)


class ConnectionPool:
    """
    LRU connection pool for managing gRPC channels to workers.

    Maintains a bounded number of connections (default 50) and evicts least-recently-used
    connections when the pool is full.
    """

    def __init__(self, max_size: int = 50):
        """
        Initialize connection pool.

        Args:
            max_size: Maximum number of connections to maintain (default 50)
        """
        self.max_size = max_size
        # OrderedDict maintains insertion order, perfect for LRU
        self._pool: OrderedDict[str, grpc.aio.Channel] = OrderedDict()
        self._lock = asyncio.Lock()
        logger.info(f"ConnectionPool initialized with max_size={max_size}")

    async def get_channel(self, worker_addr: str) -> grpc.aio.Channel:
        """
        Get or create a gRPC channel for the given worker address.

        This method is thread-safe and implements LRU eviction.

        Args:
            worker_addr: Worker address in format "host:port"

        Returns:
            gRPC channel to the worker
        """
        async with self._lock:
            # Check if connection exists (and move to end for LRU)
            if worker_addr in self._pool:
                # Move to end (most recently used)
                self._pool.move_to_end(worker_addr)
                logger.debug(f"Reusing connection to {worker_addr}")
                return self._pool[worker_addr]

            # Need to create new connection
            # If pool is full, evict least recently used
            if len(self._pool) >= self.max_size:
                evicted_addr, evicted_channel = self._pool.popitem(last=False)
                logger.info(f"Evicting LRU connection to {evicted_addr}")
                # Close evicted channel asynchronously
                await evicted_channel.close()

            # Create new channel
            logger.info(f"Creating new connection to {worker_addr} (pool size: {len(self._pool) + 1}/{self.max_size})")
            channel = grpc.aio.insecure_channel(
                worker_addr,
                options=[
                    ("grpc.max_send_message_length", 200 * 1024 * 1024),  # Match worker limit
                    ("grpc.max_receive_message_length", 200 * 1024 * 1024),  # Match worker limit
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 5000),
                    ("grpc.keepalive_permit_without_calls", 1),
                    ("grpc.http2.max_pings_without_data", 0),
                ],
            )

            self._pool[worker_addr] = channel
            return channel

    async def close_all(self):
        """
        Close all connections in the pool.

        Should be called during shutdown.
        """
        async with self._lock:
            logger.info(f"Closing all {len(self._pool)} connections")
            for addr, channel in self._pool.items():
                try:
                    await channel.close()
                    logger.debug(f"Closed connection to {addr}")
                except Exception as e:
                    logger.error(f"Error closing connection to {addr}: {e}")
            self._pool.clear()

    async def remove(self, worker_addr: str):
        """
        Remove and close a specific connection.

        Args:
            worker_addr: Worker address to remove
        """
        async with self._lock:
            if worker_addr in self._pool:
                channel = self._pool.pop(worker_addr)
                await channel.close()
                logger.info(f"Removed connection to {worker_addr}")

    def size(self) -> int:
        """Get current pool size."""
        return len(self._pool)

    async def get_stats(self) -> dict:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool stats
        """
        async with self._lock:
            return {
                "size": len(self._pool),
                "max_size": self.max_size,
                "utilization": len(self._pool) / self.max_size if self.max_size > 0 else 0,
                "connections": list(self._pool.keys()),
            }


# Global connection pool instance
_global_pool: Optional[ConnectionPool] = None


def get_connection_pool(max_size: int = 50) -> ConnectionPool:
    """
    Get the global connection pool instance (singleton pattern).

    Args:
        max_size: Maximum pool size (only used on first call)

    Returns:
        Global ConnectionPool instance
    """
    global _global_pool
    if _global_pool is None:
        _global_pool = ConnectionPool(max_size=max_size)
    return _global_pool
