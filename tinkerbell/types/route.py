"""RouteKey: typed identifier for a training/sampling engine.

Replaces the `f"{model_name}:{adapter_name}"` string scheme that the old
`GlobalStore.request_queue` keyed on. Two routes differ iff either the model
or the adapter differs; adapter=None and adapter="" are the same route.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteKey:
    model: str
    adapter: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "adapter", self.adapter or None)

    def serialize(self) -> str:
        """Stable string form (used only for logging/Ray named-actor lookups)."""
        return f"{self.model}::{self.adapter or ''}"

    @classmethod
    def parse(cls, s: str) -> "RouteKey":
        if "::" not in s:
            return cls(model=s, adapter=None)
        model, adapter = s.split("::", 1)
        return cls(model=model, adapter=adapter or None)

    def __str__(self) -> str:
        return self.serialize()
