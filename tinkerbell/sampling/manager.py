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
    """Consolidated state for a sampling actor."""

    actor: SGLangSamplingActor
    status: SamplingActorStatus
    pending_ref: Optional[Any] = None  # Ray ObjectRef for async operations


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
        """Create sampling actor. Uses model_name as key (defaults to model_id)."""
        key = model_name or model_id
        if key in self.actors:
            logger.info(f"Sampling actor for {key} already exists")
            return key

        actor_name = self._get_actor_name(key)
        logger.info(
            f"Creating sampling actor: {actor_name} (tp_size={tp_size}, model={model_id})"
        )

        actor = self._create_actor_with_options(
            model_id, actor_name, tp_size, engine_kwargs
        )
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
        """Start loading a checkpoint in the background (fire-and-forget)."""
        state = self._get_state_or_raise(model_id)
        logger.info(f"Loading checkpoint for {model_id} from {checkpoint_path}")

        state.status = SamplingActorStatus.PENDING
        state.pending_ref = state.actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
            pin_lora=pin_lora,
        )
        return True

    async def shutdown(self, model_id: str) -> bool:
        """Shutdown a sampling actor."""
        state = self._get_state_or_raise(model_id)
        result = await state.actor.shutdown.remote()
        del self.actors[model_id]
        return result

    # Private helper methods
    def _create_actor_with_options(
        self,
        model_id: str,
        actor_name: str,
        tp_size: int,
        engine_kwargs: dict[str, Any],
    ) -> SGLangSamplingActor:
        """Create and configure a Ray actor with appropriate options."""
        return SGLangSamplingActor.options(
            num_gpus=tp_size,
            get_if_exists=True,
            lifetime="detached",
            name=actor_name,
            namespace="tinkerbell",
        ).remote(
            model_id=model_id,
            tp_size=tp_size,
            engine_kwargs=engine_kwargs,
        )

    async def _check_pending_status(self, model_id: str) -> None:
        """Check if pending operation (initialization or loading) is complete."""
        state = self.actors.get(model_id)
        if state is None or state.pending_ref is None:
            return

        if self._is_actor_dead(model_id):
            self._handle_actor_failure(model_id)
            return

        try:
            ready, _ = ray.wait([state.pending_ref], num_returns=1, timeout=0)
            if ready:
                result = await state.pending_ref
                logger.info(f"✓ Pending operation completed for {model_id}: {result}")
                state.status = SamplingActorStatus.READY
                state.pending_ref = None
        except ray.exceptions.RayActorError as e:
            actor_name = self._get_actor_name(model_id)
            logger.error("=" * 80)
            logger.error(f"RayActorError: Actor {model_id} crashed")
            logger.error(f"Error: {e}")
            logger.error(f"Actor name: {actor_name}")
            logger.error(f"Check logs: ray logs {actor_name}")
            logger.error("=" * 80)
            del self.actors[model_id]
        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"Unexpected error for {model_id}: {type(e).__name__}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            del self.actors[model_id]

    def _is_actor_dead(self, model_id: str) -> bool:
        """Check if actor is in DEAD state."""
        state = self.actors.get(model_id)
        if state is None or state.actor is None:
            return False
        try:
            actor_state = ray._private.state.actors(state.actor._actor_id.hex())
            is_dead = actor_state and actor_state.get("State") == "DEAD"
            if is_dead:
                logger.error(f"Actor {model_id} state check: DEAD")
                logger.error(f"Full actor state: {actor_state}")
            return is_dead
        except Exception as e:
            logger.warning(f"Could not check actor state for {model_id}: {e}")
            return False

    def _handle_actor_failure(self, model_id: str) -> None:
        """Handle actor failure."""
        actor_name = self._get_actor_name(model_id)
        logger.error("=" * 80)
        logger.error(f"ACTOR FAILURE: {model_id}")
        logger.error(f"Actor name: {actor_name}")
        logger.error(f"Check logs: ray logs {actor_name}")
        logger.error("=" * 80)
        del self.actors[model_id]

    def _get_state_or_raise(self, model_id: str) -> ActorState:
        """Get actor state or raise ValueError if not found."""
        state = self.actors.get(model_id)
        if state is None:
            raise ValueError(f"Sampling actor for model {model_id} not found")
        return state

    @staticmethod
    def _get_actor_name(model_id: str) -> str:
        """Generate a clean actor name from model_id."""
        cleaned = model_id.replace("/", "_").replace(":", "_").lower()
        return f"sampling_actor_{cleaned}"
