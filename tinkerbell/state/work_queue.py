"""WorkQueue: per-RouteKey FIFO of Ops.

Replaces `GlobalStore.request_queue`. Drained by TrainingExecutor.

Trigger events avoid the 2s clock-cycle polling for single-stream workloads:
submitters `notify`; executors `await wait_for_work(timeout)`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray

from tinkerbell.state.ops import Op
from tinkerbell.types.route import RouteKey

logger = logging.getLogger(__name__)

WORK_QUEUE_ACTOR_NAME = "tinkerbell:work_queue"


@ray.remote(num_cpus=0)
class WorkQueue:
    def __init__(self):
        self.queues: dict[RouteKey, list[Op]] = {}
        self.events: dict[RouteKey, asyncio.Event] = {}

    def _event(self, route: RouteKey) -> asyncio.Event:
        ev = self.events.get(route)
        if ev is None:
            ev = asyncio.Event()
            self.events[route] = ev
        return ev

    async def submit(self, route: RouteKey, op: Op) -> None:
        self.queues.setdefault(route, []).append(op)
        self._event(route).set()

    async def drain(self, route: RouteKey) -> list[Op]:
        """Pop all queued ops for a route. Clears the trigger event."""
        ops = self.queues.get(route, [])
        self.queues[route] = []
        ev = self.events.get(route)
        if ev is not None:
            ev.clear()
        return ops

    async def wait_for_work(self, route: RouteKey, timeout: float) -> bool:
        """Await until ops are queued for `route` or `timeout` elapses.

        Returns True if work is ready, False on timeout.
        """
        ev = self._event(route)
        if self.queues.get(route):
            return True
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def depth(self) -> dict[str, int]:
        """Current queue depths keyed by serialized route. For diagnostics."""
        return {str(k): len(v) for k, v in self.queues.items() if v}

    async def clear(self, route: RouteKey) -> None:
        self.queues[route] = []
        ev = self.events.get(route)
        if ev is not None:
            ev.clear()


def get_or_create_work_queue() -> Any:
    return WorkQueue.options(
        num_cpus=0,
        get_if_exists=True,
        lifetime="detached",
        name=WORK_QUEUE_ACTOR_NAME,
        namespace="tinkerbell",
    ).remote()
