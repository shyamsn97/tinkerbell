"""Small resource specs passed through to Ray actor creation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActorResources:
    num_cpus: float = 1
    num_gpus: float = 1
    memory: int | None = None
    resources: dict[str, float] = field(default_factory=dict)
    accelerator_type: str | None = None

    def actor_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
        }
        if self.memory is not None:
            opts["memory"] = self.memory
        if self.resources:
            opts["resources"] = dict(self.resources)
        if self.accelerator_type is not None:
            opts["accelerator_type"] = self.accelerator_type
        return opts


__all__ = ["ActorResources"]
