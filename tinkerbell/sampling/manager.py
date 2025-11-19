from enum import Enum
from typing import Any, Dict, Optional

import ray

from tinkerbell.sampling.actor import SGLangSamplingActor


class SamplingActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    LOADING = "loading"
    NOT_SETUP = "not_present"


class SamplingManager:
    def __init__(self):
        self.sampling_actors: Dict[str, SGLangSamplingActor] = {}
        self.statuses: Dict[str, SamplingActorStatus] = {}
        self.ready_refs: Dict[str, Any] = {}
        self.loading_refs: Dict[str, Any] = {}

        if not ray.is_initialized():
            ray.init(
                address="auto",
                ignore_reinit_error=True,
                namespace="tinkerbell",
            )

    def create_sampling_actor(
        self,
        model_id: str,
        tp_size: int,
        engine_kwargs: dict[str, Any] = {},
    ) -> str:
        if model_id in self.sampling_actors:
            print(f"Sampling actor for model {model_id} already exists...")
            return model_id

        actor = self._create_actor_with_options(model_id, tp_size, engine_kwargs)
        self.sampling_actors[model_id] = actor
        self.statuses[model_id] = SamplingActorStatus.INITIALIZING
        self.ready_refs[model_id] = actor.is_ready.remote()
        return model_id

    async def get_sampling_actor_status(self, model_id: str) -> SamplingActorStatus:
        current_status = self.statuses.get(model_id, SamplingActorStatus.NOT_SETUP)

        if current_status == SamplingActorStatus.INITIALIZING:
            self._check_initializing_status(model_id)
        elif current_status == SamplingActorStatus.LOADING:
            self._check_loading_status(model_id)

        return self.statuses.get(model_id, SamplingActorStatus.NOT_SETUP)

    def get_sampling_actor(self, model_id: str) -> Optional[SGLangSamplingActor]:
        return self.sampling_actors.get(model_id)

    async def load_checkpoint(self, model_id: str, checkpoint_path: str) -> bool:
        """Start loading a checkpoint in the background (fire-and-forget)."""
        sampling_actor = self._get_actor_or_raise(model_id)

        self.statuses[model_id] = SamplingActorStatus.LOADING
        self.loading_refs[model_id] = sampling_actor.update_weights_from_disk.remote(
            checkpoint_path=checkpoint_path,
            load_format=None,
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

    def _check_initializing_status(self, model_id: str) -> None:
        """Check if initialization is complete and update status."""
        try:
            ready, _ = ray.wait([self.ready_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                ray.get(self.ready_refs[model_id])
                self.statuses[model_id] = SamplingActorStatus.READY
        except ray.exceptions.RayActorError as e:
            print("=" * 80)
            print(f"ERROR: Sampling actor {model_id} crashed during initialization!")
            print(f"RayActorError: {e}")
            print("=" * 80)
            print("The actor likely crashed due to one of these reasons:")
            print("  1. Out of memory (OOM) during model loading")
            print("  2. Model not found or inaccessible")
            print("  3. GPU out of memory")
            print("  4. SGLang server failed to start")
            print("=" * 80)
            print(
                f"To debug, check Ray logs with: ray logs {self._get_actor_name(model_id)}"
            )
            print("Or check all Ray logs at: /tmp/ray/session_latest/logs/")
            print("=" * 80)
            # Clean up the failed actor
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP
        except Exception as e:
            print("=" * 80)
            print(f"ERROR: Unexpected error during actor initialization for {model_id}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {e}")
            print("=" * 80)
            import traceback

            traceback.print_exc()
            print("=" * 80)
            # Clean up the failed actor
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP

    def _check_loading_status(self, model_id: str) -> None:
        """Check if checkpoint loading is complete and update status."""
        # First check if actor is still alive
        actor = self.sampling_actors.get(model_id)
        if actor is not None:
            try:
                # Check if actor is still alive using ray's internal state
                actor_state = ray._private.state.actors(actor._actor_id.hex())
                if actor_state and actor_state.get("State") == "DEAD":
                    print("=" * 80)
                    print(
                        f"ERROR: Sampling actor {model_id} DIED during checkpoint loading"
                    )
                    print(f"Actor state: {actor_state}")
                    print("=" * 80)
                    self._cleanup_actor_state(model_id)
                    self.statuses[model_id] = SamplingActorStatus.NOT_SETUP
                    return
            except Exception:
                # If we can't check state, continue with normal flow
                pass

        try:
            ready, _ = ray.wait([self.loading_refs[model_id]], num_returns=1, timeout=0)
            if ready:
                result = ray.get(self.loading_refs[model_id])
                print(f"Checkpoint loading completed for {model_id}: {result}")
                self.statuses[model_id] = SamplingActorStatus.READY
                del self.loading_refs[model_id]
        except ray.exceptions.RayActorError as e:
            print("=" * 80)
            print(
                f"ERROR: Sampling actor {model_id} crashed during checkpoint loading!"
            )
            print(f"RayActorError: {e}")
            print("=" * 80)
            print("The actor likely crashed due to one of these reasons:")
            print("  1. Out of memory (OOM) during checkpoint loading")
            print("  2. Checkpoint file corruption or incompatibility")
            print("  3. Model architecture mismatch")
            print("  4. GPU out of memory")
            print("=" * 80)
            print(
                f"To debug, check Ray logs with: ray logs {self._get_actor_name(model_id)}"
            )
            print("Or check all Ray logs at: /tmp/ray/session_latest/logs/")
            print("=" * 80)
            # Clean up the failed actor
            self._cleanup_actor_state(model_id)
            self.statuses[model_id] = SamplingActorStatus.NOT_SETUP
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
