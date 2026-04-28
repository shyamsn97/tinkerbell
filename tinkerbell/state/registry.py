"""Registry: single source of truth for live engines.

Replaces the three-way reconcile (`SamplingManager.actors` dict ↔
`GlobalStore.sampling_actors` ↔ `ray.get_actor(...)`) with one actor that
owns engine handles, keyed by RouteKey.

Entries are serialized (`EngineRecord`) because Ray actor handles are not
picklable across detached storage cleanly — on gateway boot we rehydrate by
doing `ray.get_actor(name)` for each record. After boot, handles are cached
in-process by the gateway and only refreshed on miss.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ray

from tinkerbell.types.route import RouteKey

logger = logging.getLogger(__name__)

REGISTRY_ACTOR_NAME = "tinkerbell:registry"
EngineKind = Literal["training", "sampling"]


@dataclass
class EngineRecord:
    """Metadata for a live engine. Worker Ray names are resolvable via ray.get_actor."""

    kind: EngineKind
    route: RouteKey
    base_model: str
    worker_names: list[str]  # Ray actor names in the "tinkerbell" namespace
    adapters: dict[str, str] = field(
        default_factory=dict
    )  # adapter_name -> checkpoint_path
    created_at: float = 0.0


@dataclass
class GroupHandle:
    """Local-cache wrapper: `EngineRecord` + resolved Ray actor handles."""

    record: EngineRecord
    workers: list[Any]  # list of ray actor handles

    @property
    def kind(self) -> EngineKind:
        return self.record.kind

    @property
    def route(self) -> RouteKey:
        return self.record.route

    @property
    def base_model(self) -> str:
        return self.record.base_model

    @property
    def adapters(self) -> dict[str, str]:
        return self.record.adapters


@ray.remote(num_cpus=0)
class Registry:
    def __init__(self):
        self.records: dict[RouteKey, EngineRecord] = {}

    async def register(self, record: EngineRecord) -> None:
        if record.created_at == 0.0:
            record.created_at = time.time()
        self.records[record.route] = record
        logger.info(
            f"Registered {record.kind} engine for route={record.route} "
            f"workers={record.worker_names}"
        )

    async def lookup(self, route: RouteKey) -> EngineRecord | None:
        return self.records.get(route)

    async def list(self, kind: EngineKind | None = None) -> list[EngineRecord]:
        records = list(self.records.values())
        if kind:
            records = [r for r in records if r.kind == kind]
        return records

    async def remove(self, route: RouteKey) -> bool:
        existed = route in self.records
        self.records.pop(route, None)
        return existed

    async def update_adapter(
        self, route: RouteKey, adapter: str, checkpoint_path: str
    ) -> None:
        rec = self.records.get(route)
        if rec is not None:
            rec.adapters[adapter] = checkpoint_path

    async def health_check(self, route: RouteKey) -> bool:
        """Ping workers for `route`; remove record if any worker is gone."""
        rec = self.records.get(route)
        if rec is None:
            return False
        try:
            for name in rec.worker_names:
                ray.get_actor(name=name, namespace="tinkerbell")
        except ValueError:
            logger.warning(f"Registry.health_check: {route} has dead worker; removing")
            self.records.pop(route, None)
            return False
        return True


def get_or_create_registry() -> Any:
    return Registry.options(
        num_cpus=0,
        get_if_exists=True,
        lifetime="detached",
        name=REGISTRY_ACTOR_NAME,
        namespace="tinkerbell",
    ).remote()


def resolve_handle(record: EngineRecord) -> Optional[GroupHandle]:
    """Resolve worker Ray handles for a record. None if any worker is unreachable."""
    try:
        workers = [
            ray.get_actor(name=n, namespace="tinkerbell") for n in record.worker_names
        ]
    except ValueError as e:
        logger.warning(f"Registry: worker lookup failed for {record.route}: {e}")
        return None
    return GroupHandle(record=record, workers=workers)
