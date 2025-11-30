import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import ray

from tinkerbell.sampling.actor import SGLangSamplingActor
from tinkerbell.store import GlobalStore

logger = logging.getLogger(__name__)


class SamplingActorStatus(Enum):
    READY = "ready"
    PENDING = "pending"
    NOT_PRESENT = "not_present"


@dataclass
class ActorState:
    actor: SGLangSamplingActor
    status: SamplingActorStatus
    pending_ref: Optional[Any] = None


class SamplingManager:
    def __init__(self, global_store: GlobalStore):
        self.actors: Dict[str, ActorState] = {}
        self.global_store = global_store

    def create_sampling_actor(
        self,
        model_id: str,
        model_name: str | None = None,
        tp_size: int = 1,
        engine_kwargs: dict[str, Any] = {},
    ) -> str:
        key = model_name or model_id
        if key in self.actors:
            return key

        actor_name = self._get_actor_name(key)
        actor = SGLangSamplingActor.options(
            num_gpus=tp_size,
            get_if_exists=True,
            lifetime="detached",
            name=actor_name,
            namespace="tinkerbell",
        ).remote(model_id=model_id, tp_size=tp_size, engine_kwargs=engine_kwargs)

        self.actors[key] = ActorState(
            actor=actor,
            status=SamplingActorStatus.PENDING,
            pending_ref=actor.is_ready.remote(),
        )
        return key

    async def get_sampling_actor_status(self, model_id: str) -> SamplingActorStatus:
        state = self.actors.get(model_id)
        if state is None:
            return SamplingActorStatus.NOT_PRESENT
        if state.status == SamplingActorStatus.PENDING:
            await self._check_pending_status(model_id)
        return self.actors.get(
            model_id, ActorState(None, SamplingActorStatus.NOT_PRESENT)
        ).status

    def get_sampling_actor(self, model_id: str) -> Optional[SGLangSamplingActor]:
        state = self.actors.get(model_id)
        return state.actor if state else None

    async def load_checkpoint(
        self, model_id: str, checkpoint_path: str, pin_lora: bool = False
    ) -> bool:
        state = self._get_state_or_raise(model_id)
        state.status = SamplingActorStatus.PENDING
        state.pending_ref = state.actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
            pin_lora=pin_lora,
        )
        return True

    async def shutdown(self, model_id: str) -> bool:
        state = self._get_state_or_raise(model_id)
        result = await state.actor.shutdown.remote()
        del self.actors[model_id]
        return result

    async def _check_pending_status(self, model_id: str) -> None:
        state = self.actors.get(model_id)
        if state is None or state.pending_ref is None:
            return

        if self._is_actor_dead(model_id):
            del self.actors[model_id]
            return

        try:
            ready, _ = ray.wait([state.pending_ref], num_returns=1, timeout=0)
            if ready:
                await state.pending_ref
                state.status = SamplingActorStatus.READY
                state.pending_ref = None
        except ray.exceptions.RayActorError as e:
            logger.error(f"Actor {model_id} crashed: {e}")
            del self.actors[model_id]
        except Exception as e:
            logger.error(f"Error for {model_id}: {e}")
            del self.actors[model_id]

    def _is_actor_dead(self, model_id: str) -> bool:
        state = self.actors.get(model_id)
        if state is None or state.actor is None:
            return False
        try:
            actor_state = ray._private.state.actors(state.actor._actor_id.hex())
            return actor_state and actor_state.get("State") == "DEAD"
        except Exception:
            return False

    def _get_state_or_raise(self, model_id: str) -> ActorState:
        state = self.actors.get(model_id)
        if state is None:
            raise ValueError(f"Sampling actor {model_id} not found")
        return state

    @staticmethod
    def _get_actor_name(model_id: str) -> str:
        cleaned = model_id.replace("/", "_").replace(":", "_").lower()
        return f"sampling_actor_{cleaned}"
