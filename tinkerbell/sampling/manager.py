import logging
import threading
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
        self._creation_lock = threading.Lock()

    def create_sampling_actor(
        self,
        base_model: str,
        model_name: str,
        tp_size: int = 1,
        engine_kwargs: dict[str, Any] = {},
    ) -> str:
        """Create a sampling actor.

        Args:
            base_model: HuggingFace model path to load (e.g., "Qwen/Qwen3-0.6B")
            model_name: Actor group name for routing (e.g., "my_model")
            tp_size: Tensor parallel size
            engine_kwargs: SGLang engine kwargs
        """
        # Use lock to prevent race condition when multiple requests
        # try to create the same actor simultaneously
        with self._creation_lock:
            if model_name in self.actors:
                logger.info(f"Sampling actor '{model_name}' already exists, reusing")
                return model_name

            actor_name = self._get_actor_name(model_name)
            logger.info(
                f"Creating sampling actor '{actor_name}' loading model '{base_model}'"
            )
            actor = SGLangSamplingActor.options(
                num_gpus=tp_size,
                get_if_exists=True,
                lifetime="detached",
                name=actor_name,
                namespace="tinkerbell",
            ).remote(
                base_model=base_model, tp_size=tp_size, engine_kwargs=engine_kwargs
            )

            self.actors[model_name] = ActorState(
                actor=actor,
                status=SamplingActorStatus.PENDING,
                pending_ref=actor.is_ready.remote(),
            )
            return model_name

    async def get_sampling_actor_status(self, model_name: str) -> SamplingActorStatus:
        """Get status of sampling actor by model_name."""
        state = self.actors.get(model_name)
        if state is None:
            return SamplingActorStatus.NOT_PRESENT
        if state.status == SamplingActorStatus.PENDING:
            await self._check_pending_status(model_name)
        return self.actors.get(
            model_name, ActorState(None, SamplingActorStatus.NOT_PRESENT)
        ).status

    def get_sampling_actor(self, model_name: str) -> Optional[SGLangSamplingActor]:
        """Get sampling actor by model_name."""
        state = self.actors.get(model_name)
        return state.actor if state else None

    async def load_checkpoint(
        self, model_name: str, checkpoint_path: str, pin_lora: bool = False
    ) -> bool:
        """Load checkpoint into sampling actor identified by model_name."""
        state = self._get_state_or_raise(model_name)
        state.status = SamplingActorStatus.PENDING
        state.pending_ref = state.actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
            pin_lora=pin_lora,
        )
        return True

    async def shutdown(self, model_name: str) -> bool:
        """Shutdown sampling actor identified by model_name."""
        state = self._get_state_or_raise(model_name)
        result = await state.actor.shutdown.remote()
        del self.actors[model_name]
        return result

    async def _check_pending_status(self, model_name: str) -> None:
        state = self.actors.get(model_name)
        if state is None or state.pending_ref is None:
            return

        if self._is_actor_dead(model_name):
            del self.actors[model_name]
            return

        try:
            ready, _ = ray.wait([state.pending_ref], num_returns=1, timeout=0)
            if ready:
                await state.pending_ref
                state.status = SamplingActorStatus.READY
                state.pending_ref = None
        except ray.exceptions.RayActorError as e:
            logger.error(f"Actor {model_name} crashed: {e}")
            del self.actors[model_name]
        except Exception as e:
            logger.error(f"Error for {model_name}: {e}")
            del self.actors[model_name]

    def _is_actor_dead(self, model_name: str) -> bool:
        state = self.actors.get(model_name)
        if state is None or state.actor is None:
            return False
        try:
            actor_state = ray._private.state.actors(state.actor._actor_id.hex())
            return actor_state and actor_state.get("State") == "DEAD"
        except Exception:
            return False

    def _get_state_or_raise(self, model_name: str) -> ActorState:
        state = self.actors.get(model_name)
        if state is None:
            raise ValueError(f"Sampling actor '{model_name}' not found")
        return state

    @staticmethod
    def _get_actor_name(model_name: str) -> str:
        """Convert model_name to Ray actor name."""
        cleaned = model_name.replace("/", "_").replace(":", "_").lower()
        return f"sampling_actor_{cleaned}"
