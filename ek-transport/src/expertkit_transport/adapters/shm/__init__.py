"""Same-host shared-memory Worker Transport."""

from expertkit_transport.adapters.shm.buffers import ShmOutputBufferProvider
from expertkit_transport.adapters.shm.client import ShmWorkerTransport
from expertkit_transport.adapters.shm.memory import (
    SharedMemoryLayout,
    SharedMemoryRegion,
    SharedMemorySlot,
)

__all__ = [
    "SharedMemoryLayout",
    "SharedMemoryRegion",
    "SharedMemorySlot",
    "ShmOutputBufferProvider",
    "ShmWorkerTransport",
]
