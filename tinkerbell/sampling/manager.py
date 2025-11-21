import logging
from enum import Enum
from typing import Any, Dict, Optional

import ray

from tinkerbell.sampling.actor import SGLangSamplingActor
from tinkerbell.store import GlobalStore

logger = logging.getLogger(__name__)


class SamplingActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    LOADING = "loading"
    NOT_SETUP = "not_present"


class SamplingManager:
    def __init__(self, global_store: GlobalStore):
        self.sampling_actors: Dict[str, SGLangSamplingActor] = {}
        self.statuses: Dict[str, SamplingActorStatus] = {}
        self.ready_refs: Dict[str, Any] = {}
        self.loading_refs: Dict[str, Any] = {}
        self.global_store = global_store

    def create_sampling_actor(
        self,
        model_id: str,
        tp_size: int,
        engine_kwargs: dict[str, Any] = {},
    ) -> str:
        if model_id in self.sampling_actors:
            logger.info(f"Sampling actor for {model_id} already exists")
            return model_id

        actor_name = self._get_actor_name(model_id)
        logger.info(f"Creating sampling actor: {actor_name} (tp_size={tp_size})")
        logger.info(f"Check logs: ray logs {actor_name}")

        actor = self._create_actor_with_options(model_id, tp_size, engine_kwargs)
        self.sampling_actors[model_id] = actor
        self.statuses[model_id] = SamplingActorStatus.INITIALIZING
        self.ready_refs[model_id] = actor.is_ready.remote()
        return model_id

    async def get_sampling_actor_status(self, model_id: str) -> SamplingActorStatus:
        current_status = self.statuses.get(model_id, SamplingActorStatus.NOT_SETUP)

        if current_status == SamplingActorStatus.INITIALIZING:
            await self._check_initializing_status(model_id)
        elif current_status == SamplingActorStatus.LOADING:
            await self._check_loading_status(model_id)

        return self.statuses.get(model_id, SamplingActorStatus.NOT_SETUP)

    def get_sampling_actor(self, model_id: str) -> Optional[SGLangSamplingActor]:
        return self.sampling_actors.get(model_id)

    async def load_checkpoint(
        self, model_id: str, checkpoint_path: str, pin_lora: bool = False
    ) -> bool:
        """Start loading a checkpoint in the background (fire-and-forget)."""
        sampling_actor = self._get_actor_or_raise(model_id)
        logger.info(f"Loading checkpoint for {model_id} from {checkpoint_path}")

        self.statuses[model_id] = SamplingActorStatus.LOADING
        self.loading_refs[model_id] = sampling_actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
            pin_lora=pin_lora,
        )
        return True

    async def shutdown(self, model_id: str) -> bool:
        """Shutdown a sampling actor."""
        sampling_actor = self._get_actor_or_raise(model_id)
        result = await sampling_actor.shutdown.remote()
        self._cleanup_actor_state(model_id)
        return result

    # Private helper methods
    def _create_actor_with_options(
        self, model_id: str, tp_size: int, engine_kwargs: dict[str, Any]
    ) -> SGLangSamplingActor:
        """Create and configure a Ray actor with appropriate options."""
        actor_name = self._get_actor_name(model_id)
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

    async def _check_initializing_status(self, model_id: str) -> None:
        """Check if initialization is complete and update status."""
        if self._is_actor_dead(model_id):
            self._handle_actor_failure(model_id, "initialization")
            return

        try:
            ready, _ = ray.wait([self.ready_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                await self.ready_refs[model_id]
                self.statuses[model_id] = SamplingActorStatus.READY
        except ray.exceptions.RayActorError as e:
            logger.error(f"Actor {model_id} crashed during initialization: {e}")
            logger.error(f"Check logs: ray logs {self._get_actor_name(model_id)}")
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP
        except Exception as e:
            logger.error(f"Unexpected error for {model_id}: {type(e).__name__}: {e}")
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP

    async def _check_loading_status(self, model_id: str) -> None:
        """Check if checkpoint loading is complete and update status."""
        if self._is_actor_dead(model_id):
            self._handle_actor_failure(model_id, "checkpoint loading")
            return
        try:
            ready, _ = ray.wait([self.loading_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                result = await self.loading_refs[model_id]
                logger.info(f"✓ Checkpoint loaded for {model_id}: {result}")
                self.statuses[model_id] = SamplingActorStatus.READY
                del self.loading_refs[model_id]
        except ray.exceptions.RayActorError as e:
            logger.error(f"Actor {model_id} crashed during checkpoint loading: {e}")
            logger.error(f"Check logs: ray logs {self._get_actor_name(model_id)}")
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP
        except Exception as e:
            logger.error(
                f"Checkpoint loading failed for {model_id}: {type(e).__name__}: {e}"
            )
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP

    def _is_actor_dead(self, model_id: str) -> bool:
        """Check if actor is in DEAD state."""
        actor = self.sampling_actors.get(model_id)
        if actor is None:
            return False
        try:
            actor_state = ray._private.state.actors(actor._actor_id.hex())
            return actor_state and actor_state.get("State") == "DEAD"
        except Exception:
            return False

    def _handle_actor_failure(self, model_id: str, phase: str) -> None:
        """Handle actor failure during initialization or loading."""
        logger.error(f"Actor {model_id} DIED during {phase}")
        logger.error(f"Check logs: ray logs {self._get_actor_name(model_id)}")
        self._cleanup_actor_state(model_id)
        self.statuses[model_id] = SamplingActorStatus.NOT_SETUP

    def _get_actor_or_raise(self, model_id: str) -> SGLangSamplingActor:
        """Get actor or raise ValueError if not found."""
        actor = self.get_sampling_actor(model_id)
        if actor is None:
            raise ValueError(f"Sampling actor for model {model_id} not found")
        return actor

    def _cleanup_actor_state(self, model_id: str) -> None:
        """Remove all tracking state for an actor."""
        self.sampling_actors.pop(model_id, None)
        self.statuses.pop(model_id, None)
        self.ready_refs.pop(model_id, None)
        self.loading_refs.pop(model_id, None)

    @staticmethod
    def _get_actor_name(model_id: str) -> str:
        """Generate a clean actor name from model_id."""
        cleaned = model_id.replace("/", "_").replace(":", "_").lower()
        return f"sampling_actor_{cleaned}"
