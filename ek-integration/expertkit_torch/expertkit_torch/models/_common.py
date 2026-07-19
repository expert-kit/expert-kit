"""Shared state used while constructing routed MoE replacement modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoutedLayerIds:
    """Hand out the real model layer IDs in construction order."""

    values: tuple[int, ...]
    _next_index: int = field(default=0, init=False)

    def take(self) -> int:
        """Return the next routed layer ID.

        Raises:
            RuntimeError: More routed blocks were constructed than the model
                configuration describes.
        """

        if self._next_index >= len(self.values):
            raise RuntimeError("model constructed more routed MoE blocks than expected")
        layer_id = self.values[self._next_index]
        self._next_index += 1
        return layer_id

    def require_complete(self) -> None:
        """Reject a model that did not construct every configured routed layer."""

        if self._next_index != len(self.values):
            raise RuntimeError(
                "model constructed "
                f"{self._next_index} routed MoE blocks, expected {len(self.values)}"
            )
