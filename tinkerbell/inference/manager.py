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
        model_path: str,
        tp_size: int,
        engine_kwargs: dict[str, Any] = {},
    ) -> str:
        if model_path in self.inference_actors:
            print(f"Inference actor for model {model_path} already exists...")
            return model_path

        cleaned_model_name = model_path.replace("/", "_").replace(":", "_").lower()
        self.inference_actors[model_path] = SGLangInferenceActor.options(
            num_gpus=tp_size,
            get_if_exists=True,
            lifetime="detached",
            name=f"inference_actor_{cleaned_model_name}",
            namespace="tinkerbell",
        ).remote(
            model_path=model_path,
            tp_size=tp_size,
            engine_kwargs=engine_kwargs,
        )
        self.statuses[model_path] = InferenceActorStatus.INITIALIZING
        self.ready_refs[model_path] = self.inference_actors[
            model_path
        ].is_ready.remote()
        return model_path

    async def get_inference_actor_status(self, model_path: str) -> InferenceActorStatus:
        # Non-blocking check
        if self.statuses[model_path] == InferenceActorStatus.INITIALIZING:
            ready, _ = ray.wait([self.ready_refs[model_path]], num_returns=1, timeout=0)
            if ready:
                self.statuses[model_path] = InferenceActorStatus.READY
                _ = ray.get(self.ready_refs[model_path])
        return self.statuses[model_path]

    def get_inference_actor(self, model_path: str) -> SGLangInferenceActor:
        return self.inference_actors.get(model_path, None)
