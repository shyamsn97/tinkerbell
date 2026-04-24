"""SamplingEngine: thin helpers around `SGLangSamplingActor`.

Unlike the training side, sampling is one actor per route — no TP fan-out at
the gateway level (SGLang handles its own TP internally). So "engine" here
is mostly a naming-for-consistency wrapper that owns the actor handle and a
readiness state.

Creation / rehydration goes through `Registry`, not a separate manager.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, Optional

import ray

from tinkerbell.sampling.actor import SGLangSamplingActor
from tinkerbell.state.registry import EngineRecord, GroupHandle
from tinkerbell.types.route import RouteKey
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


SamplingStatus = Literal["ready", "pending", "not_present"]


class SamplingEngine:
    def __init__(self, handle: GroupHandle):
        if not handle.workers:
            raise ValueError("SamplingEngine requires at least one worker")
        self.handle = handle
        self.actor = handle.workers[0]
        self.route = handle.route
        self.base_model = handle.base_model
        # Tracks the ObjectRef for in-progress init/load_checkpoint work.
        self.pending_ref: Optional[Any] = None
        self.status: SamplingStatus = "pending"

    def mark_pending(self, ref: Any) -> None:
        self.status = "pending"
        self.pending_ref = ref

    async def get_status(self) -> SamplingStatus:
        if self.status == "ready":
            return "ready"
        if self._is_actor_dead():
            self.status = "not_present"
            return self.status
        if self.pending_ref is None:
            # No outstanding work; ask the actor directly.
            try:
                is_alive = await self.actor.is_server_alive.remote()
                self.status = "ready" if is_alive else "pending"
            except Exception:
                self.status = "not_present"
            return self.status
        try:
            ready, _ = ray.wait([self.pending_ref], num_returns=1, timeout=0)
            if ready:
                await self.pending_ref
                self.status = "ready"
                self.pending_ref = None
        except ray.exceptions.RayActorError as e:
            logger.error(f"Sampling actor {self.route} died: {e}")
            self.status = "not_present"
        except Exception as e:
            logger.error(f"Sampling status check error for {self.route}: {e}")
            self.status = "not_present"
        return self.status

    def _is_actor_dead(self) -> bool:
        try:
            state = ray._private.state.actors(self.actor._actor_id.hex())
            return bool(state) and state.get("State") == "DEAD"
        except Exception:
            return False


def spawn_sampling_engine(
    *,
    base_model: str,
    route: RouteKey,
    tp_size: int = 1,
    engine_kwargs: dict[str, Any] | None = None,
) -> SamplingEngine:
    actor_name = f"sampling_actor_{clean_model_name(route.model)}"
    engine_kwargs = engine_kwargs or {}

    # Kill stale actor with the same name, if any.
    try:
        old = ray.get_actor(actor_name, namespace="tinkerbell")
        logger.warning(f"Found stale sampling actor '{actor_name}', killing it")
        ray.kill(old)
        time.sleep(1)
    except ValueError:
        pass
    except Exception as e:
        logger.warning(f"Error killing stale sampling actor: {e}")

    actor = SGLangSamplingActor.options(
        num_gpus=tp_size,
        lifetime="detached",
        name=actor_name,
        namespace="tinkerbell",
    ).remote(base_model=base_model, tp_size=tp_size, engine_kwargs=engine_kwargs)

    record = EngineRecord(
        kind="sampling",
        route=route,
        base_model=base_model,
        worker_names=[actor_name],
    )
    engine = SamplingEngine(handle=GroupHandle(record=record, workers=[actor]))
    engine.mark_pending(actor.is_ready.remote())
    return engine


def attach_sampling_engine(handle: GroupHandle) -> SamplingEngine | None:
    """Attach to an existing sampling actor, verifying it is healthy.

    Returns None if the actor is dead/unreachable.
    """
    if not handle.workers:
        return None
    actor = handle.workers[0]
    try:
        is_alive = ray.get(actor.is_server_alive.remote(), timeout=5.0)
        if not is_alive:
            logger.warning(f"Sampling actor {handle.route} has dead server")
            return None
    except Exception as e:
        logger.warning(f"Sampling actor {handle.route} unresponsive: {e}")
        return None
    engine = SamplingEngine(handle=handle)
    engine.status = "ready"
    return engine
