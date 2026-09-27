"""Worker-side Transfer Engine session and replay protection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from expertkit_transport.errors import TransportProtocolError
from expertkit_transport.transports.transfer_engine.arena import TransferArenaDescriptor


@dataclass(slots=True)
class TransferEngineSession:
    """Track one Frontend epoch and the registered arena it advertised."""

    client_epoch: str
    session_nonce: str
    target_session_id: str
    target_runtime_generation: str
    backend: str
    arena: TransferArenaDescriptor
    _last_generations: list[int] = field(init=False, repr=False)
    _active_sequences: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _active_slots: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _close_prepared: bool = field(default=False, init=False, repr=False)
    _idle_event: asyncio.Event = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not self.client_epoch
            or not self.session_nonce
            or not self.target_session_id
            or not self.target_runtime_generation
        ):
            raise ValueError("Transfer Engine session identifiers must not be empty")
        if not self.backend:
            raise ValueError("Transfer Engine session backend must not be empty")
        self._last_generations = [0] * self.arena.slot_count
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    @property
    def idle(self) -> bool:
        return not self._active_sequences

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def close_prepared(self) -> bool:
        return self._close_prepared

    def claim(self, *, sequence: int, slot_index: int, generation: int) -> None:
        if self._closing:
            raise TransportProtocolError("Transfer Engine session is closing")
        if not 0 <= slot_index < self.arena.slot_count:
            raise TransportProtocolError("Transfer Engine client slot is out of range")
        if sequence in self._active_sequences:
            raise TransportProtocolError("Transfer Engine request sequence is already active")
        if slot_index in self._active_slots:
            raise TransportProtocolError("Transfer Engine client slot is already active")
        if generation <= self._last_generations[slot_index]:
            raise TransportProtocolError("Transfer Engine client slot generation is stale")
        self._last_generations[slot_index] = generation
        self._active_sequences[sequence] = slot_index
        self._active_slots[slot_index] = sequence
        self._idle_event.clear()

    def release(self, sequence: int) -> None:
        if sequence not in self._active_sequences:
            raise RuntimeError("Transfer Engine session request is not active")
        slot_index = self._active_sequences.pop(sequence)
        owner = self._active_slots.pop(slot_index, None)
        if owner != sequence:
            raise RuntimeError("Transfer Engine session active-slot ownership is corrupted")
        if self.idle:
            self._idle_event.set()

    async def wait_idle(self) -> None:
        await self._idle_event.wait()

    def begin_close_prepare(self) -> bool:
        self._closing = True
        return self.idle

    def finish_close_prepare(self) -> None:
        if not self._closing or not self.idle:
            raise RuntimeError("Transfer Engine close prepare requires a drained session")
        self._close_prepared = True


__all__ = ["TransferEngineSession"]
