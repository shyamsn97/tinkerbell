from enum import Enum
from typing import Any, Dict

import ray

from tinkerbell.inference.actor import SGLangInferenceActor


class InferenceActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    NOT_SETUP = "not_setup"


class InferenceManager:
    def __init__(
        self,
    ):
        self.inference_actors: Dict[str, SGLangInferenceActor] = {}
        self.statuses: Dict[str, InferenceActorStatus] = {}
        self.ready_refs: Dict[str, Any] = {}

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
        ).remote(
            model_id=model_id,
            tp_size=tp_size,
            engine_kwargs=engine_kwargs,
        )
        self.statuses[model_id] = InferenceActorStatus.INITIALIZING
        self.ready_refs[model_id] = self.inference_actors[model_id].is_ready.remote()
        return model_id

    async def get_inference_actor_status(self, model_id: str) -> InferenceActorStatus:
        # Non-blocking check
        if self.statuses[model_id] == InferenceActorStatus.INITIALIZING:
            ready, _ = ray.wait([self.ready_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                self.statuses[model_id] = InferenceActorStatus.READY
                _ = ray.get(self.ready_refs[model_id])
        return self.statuses[model_id]

    def get_inference_actor(self, model_id: str) -> SGLangInferenceActor:
        return self.inference_actors.get(model_id, None)
