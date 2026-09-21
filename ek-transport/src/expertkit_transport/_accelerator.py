"""Transport-private access to PyTorch accelerator stream and event APIs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import cast

import torch

_SUPPORTED_ACCELERATORS = frozenset({"cuda", "npu"})


@dataclass(frozen=True, slots=True)
class TorchAccelerator:
    """Resolve the matching PyTorch device module without owning Worker runtime state."""

    device: torch.device

    def current_stream(self) -> torch.Stream:
        """Return the current stream for this accelerator and device."""

        module = torch.get_device_module(self.device)
        return cast(torch.Stream, module.current_stream(self.device))

    def create_event(self) -> torch.Event:
        """Create a dependency-only event for this accelerator."""

        module = torch.get_device_module(self.device)
        return cast(
            torch.Event,
            module.Event(enable_timing=False, blocking=False),
        )

    def stream_context(self, stream: torch.Stream) -> AbstractContextManager[None]:
        """Select one stream through its device module for the duration of a copy."""

        module = torch.get_device_module(self.device)
        return cast(AbstractContextManager[None], module.stream(stream))


def accelerator_for(device: torch.device | str) -> TorchAccelerator | None:
    """Return a transport-local adapter for CUDA/NPU, or ``None`` for CPU."""

    resolved = torch.device(device)
    if resolved.type == "cpu":
        return None
    if resolved.type not in _SUPPORTED_ACCELERATORS:
        raise ValueError("transport device must be CPU, CUDA, or NPU")
    return TorchAccelerator(resolved)
