from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fastapi import HTTPException

from tinkerbell.store import GlobalStore
from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.lora_config import LoraConfig
from tinkerbell.types.requests import ForwardBackwardRequest, ForwardRequest
from tinkerbell.types.responses import RemoteFuture
from tinkerbell.utils import get_free_port

logger = logging.getLogger(__name__)


class ActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    NOT_PRESENT = "not_present"


class ActorGroup:
    def __init__(
        self,
        workers: list[Any],
        model_id: str,
        status: ActorStatus = ActorStatus.INITIALIZING,
        max_wait_time: float = 600.0,
    ):
        self.workers = workers
        self.setup_refs = [worker.setup.remote() for worker in self.workers]
        self.model_id = model_id
        self.status = status
        self.max_wait_time = max_wait_time

    async def get_status(self) -> ActorStatus:
        if self.status == ActorStatus.INITIALIZING:
            await self._check_initialization_complete()
        return self.status

    async def wait_until_ready(self) -> bool:
        """Wait until the actor is ready. Returns True if ready, False on timeout."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < self.max_wait_time:
            if await self.get_status() == ActorStatus.READY:
                return True
            await asyncio.sleep(1.0)
        return False

    async def zero_grad(self) -> None:
        """Zero the gradients for all workers."""
        refs = [worker.zero_grad.remote() for worker in self.workers]
        await asyncio.gather(*refs)

    async def forward_backward(
        self,
        data: list[Any],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Forward and backward pass through the model with list of Datum objects."""
        await self._ensure_ready()
        outputs = await self._execute_forward_backward(
            data, forward_kwargs, return_logprobs
        )
        return self._restructure_outputs(outputs, len(data))

    async def optim_step(self, optimizer_params: dict[str, Any] = {}) -> None:
        """Step the optimizer for all workers."""
        refs = [
            worker.optim_step.remote(optimizer_params=optimizer_params)
            for worker in self.workers
        ]
        await asyncio.gather(*refs)

    async def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save the checkpoint for all workers."""
        refs = [
            worker.save_checkpoint.remote(checkpoint_path=checkpoint_path)
            for worker in self.workers
        ]
        await asyncio.gather(*refs)

    # Private helper methods
    async def _check_initialization_complete(self) -> None:
        """Check if all workers have completed initialization."""
        ready, _ = ray.wait(
            self.setup_refs, num_returns=len(self.setup_refs), timeout=0
        )
        if len(ready) == len(self.setup_refs):
            # Await the setup refs to check for any exceptions during initialization
            await asyncio.gather(*[ref for ref in self.setup_refs])
            self.status = ActorStatus.READY

    async def _ensure_ready(self) -> None:
        """Ensure actors are ready, raise HTTPException if not."""
        if self.status != ActorStatus.READY:
            if not await self.wait_until_ready():
                raise HTTPException(
                    status_code=503,
                    detail=f"Actors for model {self.model_id} are not ready",
                )

    async def _execute_forward_backward(
        self,
        data: list[Any],
        forward_kwargs: dict[str, Any],
        return_logprobs: bool,
    ) -> list[dict[str, Any]]:
        """Execute forward-backward pass on all workers."""
        refs = [
            worker.forward_backward.remote(
                data=data,
                forward_kwargs=forward_kwargs,
                return_logprobs=return_logprobs,
            )
            for worker in self.workers
        ]
        return await asyncio.gather(*refs)

    @staticmethod
    def _restructure_outputs(
        outputs: list[dict[str, Any]], batch_size: int
    ) -> list[dict[str, Any]]:
        """Restructure outputs from workers to per-sample format."""
        rank_0_output = [output for output in outputs if output is not None][0]
        batched_output = [{} for _ in range(batch_size)]
        for i in range(batch_size):
            for key in rank_0_output:
                batched_output[i][key] = rank_0_output[key][i]
        return batched_output

    @classmethod
    def create_actor_group(
        cls,
        world_size: int,
        model_id: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        scheduler_params: dict[str, Any],
        ray_worker_options: dict[str, Any] = {},
        lora_config: Optional[dict[str, Any]] = None,
        initialize_random_weights: bool = False,
        max_wait_time: float = 600.0,
    ) -> ActorGroup:
        """Create all training actors for the group."""
        master_addr = "127.0.0.1"
        master_port = str(get_free_port())
        cleaned_name = cls._clean_actor_name(model_id)

        workers = []
        for rank in range(world_size):
            worker = TrainingActor.options(
                num_gpus=1,
                get_if_exists=True,
                lifetime="detached",
                name=f"training_actor_{cleaned_name}_{rank}",
                namespace="tinkerbell",
                **ray_worker_options,
            ).remote(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                model_id=model_id,
                model_kwargs=model_kwargs,
                parallelize_plan=parallelize_plan,
                scheduler_params=scheduler_params,
                lora_config=lora_config,
                initialize_random_weights=initialize_random_weights,
            )
            workers.append(worker)

        return cls(
            workers=workers,
            model_id=model_id,
            status=ActorStatus.INITIALIZING,
            max_wait_time=max_wait_time,
        )

    @staticmethod
    def _clean_actor_name(model_id: str) -> str:
        """Generate a clean actor name from model_id."""
        return model_id.replace("/", "_").replace(":", "_").lower()

    @classmethod
    def _try_reconnect_to_existing_actors(
        cls, model_id: str, max_wait_time: float = 600.0
    ) -> Optional[ActorGroup]:
        """Try to reconnect to existing detached training actors by searching Ray.

        Returns ActorGroup if successfully reconnected, None otherwise.
        """
        try:
            cleaned_name = cls._clean_actor_name(model_id)
            prefix = f"training_actor_{cleaned_name}_"

            # Find all actors with matching name pattern
            all_actors = ray.util.list_named_actors(all_namespaces=True)

            # Handle different return formats from ray.util.list_named_actors()
            matching_actors = []
            for actor in all_actors:
                # If actor is a string, it's just the actor name
                if isinstance(actor, str):
                    if actor.startswith(prefix):
                        matching_actors.append(
                            {"name": actor, "namespace": "tinkerbell"}
                        )
                # If actor is a dict, use it directly
                elif isinstance(actor, dict):
                    if (
                        actor.get("name", "").startswith(prefix)
                        and actor.get("namespace") == "tinkerbell"
                    ):
                        matching_actors.append(actor)

            if not matching_actors:
                return None

            # Reconnect to actors sorted by rank
            workers = []
            for actor_info in sorted(matching_actors, key=lambda x: x["name"]):
                worker = ray.get_actor(actor_info["name"], namespace="tinkerbell")
                workers.append(worker)

            logger.info(
                f"Reconnected to {len(workers)} existing training actors for model {model_id}"
            )
            return cls(
                workers=workers,
                model_id=model_id,
                status=ActorStatus.READY,
                max_wait_time=max_wait_time,
            )
        except Exception as e:
            logger.error(f"Failed to reconnect to existing actors: {e}")
            import traceback

            traceback.print_exc()
            return None


class TrainingManager:
    def __init__(
        self,
        max_wait_time: float = 600.0,
        clock_cycle: float = 5.0,
    ):
        self.actor_groups: Dict[str, ActorGroup] = {}
        self.max_wait_time = max_wait_time
        self.clock_cycle = clock_cycle
        self.global_store = self._create_global_store()
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start the batch processor background task."""
        if self.running:
            return
        self.running = True
        if self.clock_cycle > 0.0:
            self.batch_processor_task = asyncio.create_task(
                self._batch_processor_loop()
            )

    async def stop(self):
        """Stop the batch processor background task."""
        self.running = False
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

    async def get_result(
        self, request_id: str, max_wait_time: float = 300.0
    ) -> Dict[str, Any]:
        """Get the result of a forward-backward request."""
        if not await self._wait_for_result(request_id, max_wait_time):
            raise TimeoutError(
                f"Result for request {request_id} not available after {max_wait_time} seconds"
            )
        return ray.get(self.global_store.get_result.remote(request_id=request_id))

    async def get_actor_status(self, model_id: str) -> ActorStatus:
        """Check if training actors are ready."""
        if model_id not in self.actor_groups:
            return ActorStatus.NOT_PRESENT
        return await self.actor_groups[model_id].get_status()

    async def forward(
        self,
        model_id: str,
        data: list[Any],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """Queue a forward request to be processed in the next batch."""
        request = ForwardRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            data=data,
            forward_kwargs=forward_kwargs,
        )
        return await self._queue_and_process(request, model_id)

    async def forward_backward(
        self,
        model_id: str,
        data: list[Any],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """Queue a forward-backward request to be processed in the next batch."""
        request = ForwardBackwardRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            data=data,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )
        return await self._queue_and_process(request, model_id)

    async def zero_grad(self, model_id: str) -> None:
        """Zero the gradients for all workers."""
        actor_group = self._get_actor_group_or_raise(model_id)
        await actor_group.zero_grad()

    async def optim_step(
        self, model_id: str, optimizer_params: dict[str, Any] = {}
    ) -> None:
        """Step the optimizer for all workers."""
        actor_group = self._get_actor_group_or_raise(model_id)
        await actor_group.optim_step(optimizer_params=optimizer_params)

    async def save_checkpoint(self, model_id: str, checkpoint_path: str) -> None:
        """Save the checkpoint for all workers."""
        actor_group = self._get_actor_group_or_raise(model_id)
        await actor_group.save_checkpoint(checkpoint_path=checkpoint_path)

    async def create_training_actors(
        self,
        world_size: int,
        model_id: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        scheduler_params: dict[str, Any],
        ray_worker_options: dict[str, Any] = {},
        lora_config: Optional[LoraConfig | dict[str, Any]] = None,
        initialize_random_weights: bool = False,
    ) -> str:
        """Create a training worker for the given model id."""
        await self.start()
        if model_id in self.actor_groups:
            return model_id

        # Try to reconnect to existing actors first
        try:
            existing_group = self._get_actor_group_or_raise(model_id)
            self.actor_groups[model_id] = existing_group
            return model_id
        except Exception:
            pass

        # Parse LoRA config if provided as dict
        lora_config_dict = None
        if lora_config is not None:
            if isinstance(lora_config, dict):
                lora_config_dict = lora_config
            else:
                lora_config_dict = lora_config.model_dump()

        # Create new actors if reconnection failed
        self.actor_groups[model_id] = ActorGroup.create_actor_group(
            world_size=world_size,
            model_id=model_id,
            model_kwargs=model_kwargs,
            parallelize_plan=parallelize_plan,
            scheduler_params=scheduler_params,
            ray_worker_options=ray_worker_options,
            lora_config=lora_config_dict,
            initialize_random_weights=initialize_random_weights,
            max_wait_time=self.max_wait_time,
        )

        return model_id

    # Private helper methods

    def _get_actor_group_or_raise(self, model_id: str) -> ActorGroup:
        """Get actor group or raise ValueError if not found."""
        if model_id not in self.actor_groups:
            # Try to reconnect to existing actors before raising
            existing_group = ActorGroup._try_reconnect_to_existing_actors(
                model_id, max_wait_time=self.max_wait_time
            )
            if existing_group:
                self.actor_groups[model_id] = existing_group
                return existing_group

            raise ValueError(
                f"Training actors for model '{model_id}' not found. "
                f"Please create training actors first using create_training_actors(). "
                f"Available models: {list(self.actor_groups.keys())}"
            )
        return self.actor_groups[model_id]

    @staticmethod
    def _create_global_store() -> Any:
        """Create the global store Ray actor."""
        return GlobalStore.options(
            num_gpus=0,
            get_if_exists=True,
            lifetime="detached",
            name="tinkerbell_global_state_manager",
            namespace="tinkerbell",
        ).remote()

    async def _wait_for_result(self, request_id: str, max_wait_time: float) -> bool:
        """Poll for result availability. Returns True if found, False on timeout."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            results = ray.get(self.global_store.get_results.remote())
            if request_id in results:
                return True
            await asyncio.sleep(1.0)
        return False

    async def _queue_and_process(
        self, request: ForwardRequest | ForwardBackwardRequest, model_id: str
    ) -> RemoteFuture:
        """Queue a request and optionally process immediately."""
        ray.get(self.global_store.add_request_to_queue.remote(request=request))
        if self.clock_cycle <= 0.0:
            await self._process_batch(model_id)
        return RemoteFuture(request_id=request.request_id)

    async def _batch_processor_loop(self) -> None:
        """Background task that processes batches at regular intervals."""
        while self.running:
            try:
                await asyncio.sleep(self.clock_cycle)
                await self._process_all_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.stop()
                raise e

    async def _process_all_batches(self) -> None:
        """Process all pending batches across all model groups."""
        request_queue = ray.get(self.global_store.get_request_queue.remote())
        for model_id in list(request_queue.keys()):
            if len(request_queue[model_id]) > 0:
                await self._process_batch(model_id)

    async def _process_batch(self, model_id: str) -> None:
        """Process a batch of requests for a specific model."""
        requests = self._get_pending_requests(model_id)
        if not requests:
            return

        self.global_store.clear_request_queue.remote(model_id=model_id)

        if await self.get_actor_status(model_id) != ActorStatus.READY:
            return

        batch_data, request_sizes = self._prepare_batch_data(requests)
        output = await self._execute_batch(model_id, requests[0], batch_data)
        self._distribute_results(requests, output, request_sizes)

    def _get_pending_requests(self, model_id: str) -> list[Any]:
        """Get all pending requests for a model."""
        request_queue = ray.get(self.global_store.get_request_queue.remote())
        return request_queue.get(model_id, [])

    @staticmethod
    def _prepare_batch_data(requests: list[Any]) -> tuple[list[Any], list[int]]:
        """Flatten request data into a single batch and track sizes."""
        batch_data = []
        request_sizes = []
        for req in requests:
            request_sizes.append(len(req.data))
            batch_data.extend(req.data)
        return batch_data, request_sizes

    async def _execute_batch(
        self, model_id: str, sample_request: Any, batch_data: list[Any]
    ) -> list[dict[str, Any]]:
        """Execute the batched forward-backward pass."""
        actor_group = self._get_actor_group_or_raise(model_id)
        return await actor_group.forward_backward(
            data=batch_data,
            forward_kwargs=sample_request.forward_kwargs,
            return_logprobs=getattr(sample_request, "return_logprobs", False),
        )

    def _distribute_results(
        self,
        requests: list[Any],
        output: list[dict[str, Any]],
        request_sizes: list[int],
    ) -> None:
        """Distribute batch results back to individual requests."""
        output_idx = 0
        for req, req_size in zip(requests, request_sizes):
            req_outputs = output[output_idx : output_idx + req_size]
            output_idx += req_size

            if req_outputs:
                combined_output = self._combine_request_outputs(req_outputs)
                self.global_store.set_result.remote(
                    request_id=req.request_id, result=combined_output
                )

    @staticmethod
    def _combine_request_outputs(
        req_outputs: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        """Combine outputs for a single request into a dict of lists."""
        combined_output = {}
        for key in req_outputs[0].keys():
            combined_output[key] = [out[key] for out in req_outputs]
        return combined_output
