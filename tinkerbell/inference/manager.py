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
    def __init__(
        self,
    ):
        self.inference_actors: Dict[str, SGLangInferenceActor] = {}
        self.statuses: Dict[str, InferenceActorStatus] = {}
        self.ready_refs: Dict[str, Any] = {}
        self.loading_refs: Dict[str, Any] = {}  # Track checkpoint loading

        if not ray.is_initialized():
            ray.init(
                address="auto",  # This connects to existing cluster
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

        cleaned_model_name = model_id.replace("/", "_").replace(":", "_").lower()
        self.inference_actors[model_id] = SGLangInferenceActor.options(
            num_gpus=tp_size,
            get_if_exists=True,
            lifetime="detached",
            name=f"inference_actor_{cleaned_model_name}",
            namespace="tinkerbell",
            max_concurrency=100,  # Allow concurrent requests for SGLang batching
        ).remote(
            model_id=model_id,
            tp_size=tp_size,
            engine_kwargs=engine_kwargs,
        )
        self.statuses[model_id] = InferenceActorStatus.INITIALIZING
        self.ready_refs[model_id] = self.inference_actors[model_id].is_ready.remote()
        return model_id

    async def get_inference_actor_status(self, model_id: str) -> InferenceActorStatus:
        # Non-blocking check - FastAPI/Ray Serve already manages the event loop
        current_status = self.statuses.get(model_id, InferenceActorStatus.NOT_SETUP)

        if current_status == InferenceActorStatus.INITIALIZING:
            ready, _ = ray.wait([self.ready_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                self.statuses[model_id] = InferenceActorStatus.READY
                _ = ray.get(self.ready_refs[model_id])

        elif current_status == InferenceActorStatus.LOADING:
            # Check if checkpoint loading is complete
            ready, _ = ray.wait([self.loading_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                # Loading complete, mark as ready
                _ = ray.get(self.loading_refs[model_id])
                self.statuses[model_id] = InferenceActorStatus.READY
                del self.loading_refs[model_id]
            return self.statuses[model_id]

        return self.statuses.get(model_id, InferenceActorStatus.NOT_SETUP)

    def get_inference_actor(self, model_id: str) -> Optional[SGLangInferenceActor]:
        return self.inference_actors.get(model_id, None)

    async def load_checkpoint(self, model_id: str, checkpoint_path: str) -> bool:
        """Start loading a checkpoint in the background (fire-and-forget)."""
        inference_actor = self.get_inference_actor(model_id)
        if inference_actor is None:
            raise ValueError(f"Inference actor for model {model_id} not found")

        # Start loading in background and return immediately
        self.statuses[model_id] = InferenceActorStatus.LOADING
        self.loading_refs[model_id] = inference_actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
        )
        return True

    async def shutdown(self, model_id: str) -> bool:
        """Shutdown an inference actor."""
        inference_actor = self.get_inference_actor(model_id)
        if inference_actor is None:
            raise ValueError(f"Inference actor for model {model_id} not found")

        result = await inference_actor.shutdown.remote()
        # Remove from tracking
        self.inference_actors.pop(model_id, None)
        self.statuses.pop(model_id, None)
        self.ready_refs.pop(model_id, None)
        return result
