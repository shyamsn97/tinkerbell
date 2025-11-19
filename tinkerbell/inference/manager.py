from enum import Enum
from typing import Any, Dict, Optional

import ray

from tinkerbell.inference.actor import SGLangInferenceActor


class InferenceActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    LOADING = "loading"
    NOT_SETUP = "not_present"


class InferenceManager:
    def __init__(self):
        self.inference_actors: Dict[str, SGLangInferenceActor] = {}
        self.statuses: Dict[str, InferenceActorStatus] = {}
        self.ready_refs: Dict[str, Any] = {}
        self.loading_refs: Dict[str, Any] = {}

        if not ray.is_initialized():
            ray.init(
                address="auto",
                ignore_reinit_error=True,
                namespace="tinkerbell",
            )

    def create_inference_actor(
        self,
        model_id: str,
        tp_size: int,
        engine_kwargs: dict[str, Any] = {},
    ) -> str:
        if model_id in self.inference_actors:
            print(f"Inference actor for model {model_id} already exists...")
            return model_id

        actor = self._create_actor_with_options(model_id, tp_size, engine_kwargs)
        self.inference_actors[model_id] = actor
        self.statuses[model_id] = InferenceActorStatus.INITIALIZING
        self.ready_refs[model_id] = actor.is_ready.remote()
        return model_id

    async def get_inference_actor_status(self, model_id: str) -> InferenceActorStatus:
        current_status = self.statuses.get(model_id, InferenceActorStatus.NOT_SETUP)

        if current_status == InferenceActorStatus.INITIALIZING:
            self._check_initializing_status(model_id)
        elif current_status == InferenceActorStatus.LOADING:
            self._check_loading_status(model_id)

        return self.statuses.get(model_id, InferenceActorStatus.NOT_SETUP)

    def get_inference_actor(self, model_id: str) -> Optional[SGLangInferenceActor]:
        return self.inference_actors.get(model_id)

    async def load_checkpoint(self, model_id: str, checkpoint_path: str) -> bool:
        """Start loading a checkpoint in the background (fire-and-forget)."""
        inference_actor = self._get_actor_or_raise(model_id)

        self.statuses[model_id] = InferenceActorStatus.LOADING
        self.loading_refs[model_id] = inference_actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
        )
        return True

    async def shutdown(self, model_id: str) -> bool:
        """Shutdown an inference actor."""
        inference_actor = self._get_actor_or_raise(model_id)
        result = await inference_actor.shutdown.remote()
        self._cleanup_actor_state(model_id)
        return result

    # Private helper methods
    def _create_actor_with_options(
        self, model_id: str, tp_size: int, engine_kwargs: dict[str, Any]
    ) -> SGLangInferenceActor:
        """Create and configure a Ray actor with appropriate options."""
        actor_name = self._get_actor_name(model_id)
        return SGLangInferenceActor.options(
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

    def _check_initializing_status(self, model_id: str) -> None:
        """Check if initialization is complete and update status."""
        ready, _ = ray.wait([self.ready_refs[model_id]], num_returns=1, timeout=0)
        if ready:
            self.statuses[model_id] = InferenceActorStatus.READY
            ray.get(self.ready_refs[model_id])

    def _check_loading_status(self, model_id: str) -> None:
        """Check if checkpoint loading is complete and update status."""
        try:
            ready, _ = ray.wait([self.loading_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                result = ray.get(self.loading_refs[model_id])
                print(f"Checkpoint loading completed for {model_id}: {result}")
                self.statuses[model_id] = InferenceActorStatus.READY
                del self.loading_refs[model_id]
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Checkpoint loading failed for {model_id}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            print("=" * 80)
            import traceback

            traceback.print_exc()
            print("=" * 80)
            # Clean up the failed actor
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = InferenceActorStatus.NOT_SETUP

    def _get_actor_or_raise(self, model_id: str) -> SGLangInferenceActor:
        """Get actor or raise ValueError if not found."""
        actor = self.get_inference_actor(model_id)
        if actor is None:
            raise ValueError(f"Inference actor for model {model_id} not found")
        return actor

    def _cleanup_actor_state(self, model_id: str) -> None:
        """Remove all tracking state for an actor."""
        self.inference_actors.pop(model_id, None)
        self.statuses.pop(model_id, None)
        self.ready_refs.pop(model_id, None)
        self.loading_refs.pop(model_id, None)

    @staticmethod
    def _get_actor_name(model_id: str) -> str:
        """Generate a clean actor name from model_id."""
        cleaned = model_id.replace("/", "_").replace(":", "_").lower()
        return f"inference_actor_{cleaned}"
