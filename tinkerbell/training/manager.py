import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fastapi import HTTPException

from tinkerbell.store import GlobalStore
from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.requests import ForwardBackwardRequest, ForwardRequest
from tinkerbell.types.responses import RemoteFuture
from tinkerbell.utils import get_free_port


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
        # Non-blocking check
        if self.status == ActorStatus.INITIALIZING:
            ready, _ = ray.wait(
                self.setup_refs, num_returns=len(self.setup_refs), timeout=0
            )
            if len(ready) == len(self.setup_refs):
                self.status = ActorStatus.READY
                _ = ray.get(self.setup_refs)
        return self.status

    async def wait_until_ready(self) -> bool:
        """
        Wait until the actor is ready.

        Returns:
            True if actor becomes ready, False if timeout occurs (max_wait_time is reached)
        """
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < self.max_wait_time:
            status = await self.get_status()
            if status == ActorStatus.READY:
                return True
            await asyncio.sleep(1.0)  # Wait 1 second before checking again
        return False

    async def zero_grad(self) -> None:
        """Zero the gradients for all workers."""
        refs = [worker.zero_grad.remote() for worker in self.workers]
        await asyncio.gather(*refs)

    async def forward_backward(
        self,
        inputs: list[dict[str, Any]],
        targets: Any = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Forward and backward pass through the model with batched inputs."""
        if self.status != ActorStatus.READY:
            if not await self.wait_until_ready():
                raise HTTPException(
                    status_code=503,
                    detail=f"Actors for model {self.model_id} are not ready",
                )
        batch_size = len(inputs)
        refs = [
            worker.forward_backward.remote(
                inputs=inputs,
                targets=targets,
                forward_kwargs=forward_kwargs,
                return_logprobs=return_logprobs,
            )
            for worker in self.workers
        ]
        outputs = await asyncio.gather(*refs)
        rank_0_output = [output for output in outputs if output is not None][0]
        batched_output = [{} for _ in range(batch_size)]
        for i in range(batch_size):
            for key in rank_0_output:
                batched_output[i][key] = rank_0_output[key][i]
        return batched_output

    async def optim_step(self, optimizer_params: dict[str, Any] = {}) -> None:
        """Step the optimizer for all workers."""
        refs = [
            worker.optim_step.remote(optimizer_params=optimizer_params)
            for worker in self.workers
        ]
        _ = await asyncio.gather(*refs)
        return None

    async def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save the checkpoint for all workers."""
        refs = [
            worker.save_checkpoint.remote(checkpoint_path=checkpoint_path)
            for worker in self.workers
        ]
        _ = await asyncio.gather(*refs)
        return None


class TrainingManager:
    def __init__(
        self,
        max_wait_time: float = 300.0,
        clock_cycle: float = 5.0,
    ):
        self.actor_groups: Dict[str, ActorGroup] = {}
        self.max_wait_time = max_wait_time
        self.clock_cycle = clock_cycle

        # Queue for batching requests
        self.global_store = GlobalStore.options(
            num_gpus=0,
            get_if_exists=True,
            lifetime="detached",
            name="tinkerbell_global_state_manager",
            namespace="tinkerbell",
        ).remote()
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start the batch processor background task."""
        if self.running:
            return
        self.running = True
        if self.clock_cycle > 0.0:
            # start the batch processor task
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
        # Wait for the result to be available
        start_time = time.time()
        elapsed_time = 0.0
        request_id_is_in_results = False
        while not request_id_is_in_results and elapsed_time < max_wait_time:
            results = ray.get(self.global_store.get_results.remote())
            if request_id in results:
                request_id_is_in_results = True
                break
            await asyncio.sleep(1.0)
            elapsed_time = time.time() - start_time
        if not request_id_is_in_results:
            raise TimeoutError(
                f"Result for request {request_id} not available after {max_wait_time} seconds"
            )
        return ray.get(self.global_store.get_result.remote(request_id=request_id))

    async def _batch_processor_loop(self) -> None:
        """Background task that processes batches at regular intervals."""
        while self.running:
            try:
                await asyncio.sleep(self.clock_cycle)
                await self._process_all_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # print(f"Error in batch processor: {e}")
                # tb_str = traceback.format_exc()
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
        # Get all pending requests
        request_queue = ray.get(self.global_store.get_request_queue.remote())
        requests = request_queue.get(model_id, [])
        if len(requests) == 0:
            return

        self.global_store.clear_request_queue.remote(model_id=model_id)
        # Check if actor is ready
        status = await self.get_actor_status(model_id)
        if status != ActorStatus.READY:
            return

        # Batch all inputs together
        batch_inputs = [req.inputs for req in requests]
        batch_targets = [req.targets for req in requests]
        # Assuming all requests use the same forward_kwargs (or merge them)
        forward_kwargs = requests[0].forward_kwargs if requests else {}
        return_logprobs = requests[0].return_logprobs if requests else False

        # Execute batch
        output = await self.actor_groups[model_id].forward_backward(
            inputs=batch_inputs,
            targets=batch_targets,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )
        # Distribute results back to futures

        for req, output in zip(requests, output, strict=True):
            if output is not None:
                self.global_store.set_result.remote(
                    request_id=req.request_id, result=output
                )

    async def get_actor_status(self, model_id: str) -> ActorStatus:
        """Check if training actors are ready."""
        if model_id not in self.actor_groups:
            return ActorStatus.NOT_PRESENT

        status = await self.actor_groups[model_id].get_status()
        return status

    async def forward(
        self,
        model_id: str,
        inputs: dict[str, Any],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """
        Queue a forward request to be processed in the next batch.
        Returns immediately with a future that will resolve when the batch is processed.
        """

        request = ForwardRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )
        _ = ray.get(self.global_store.add_request_to_queue.remote(request=request))
        if self.clock_cycle <= 0.0:
            # act immediately
            await self._process_batch(model_id)
        return RemoteFuture(request_id=request.request_id)

    async def forward_backward(
        self,
        model_id: str,
        inputs: dict[str, Any],
        targets: Any = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """
        Queue a forward-backward request to be processed in the next batch.
        Returns immediately with a future that will resolve when the batch is processed.
        """

        request = ForwardBackwardRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            inputs=inputs,
            targets=targets,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )

        # Add to queue
        _ = ray.get(self.global_store.add_request_to_queue.remote(request=request))

        if self.clock_cycle <= 0.0:
            # act immediately
            await self._process_batch(model_id)
        return RemoteFuture(request_id=request.request_id)

    async def zero_grad(self, model_id: str) -> None:
        """Zero the gradients for all workers."""
        await self.actor_groups[model_id].zero_grad()

    async def optim_step(
        self, model_id: str, optimizer_params: dict[str, Any] = {}
    ) -> None:
        """Step the optimizer for all workers."""
        await self.actor_groups[model_id].optim_step(optimizer_params=optimizer_params)

    async def save_checkpoint(self, model_id: str, checkpoint_path: str) -> None:
        """Save the checkpoint for all workers."""
        await self.actor_groups[model_id].save_checkpoint(
            checkpoint_path=checkpoint_path
        )

    async def create_training_actors(
        self,
        world_size: int,
        model_id: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        scheduler_params: dict[str, Any],
        ray_worker_options: dict[str, Any] = {},
    ) -> str:
        """Create a training worker for the given model id."""
        if model_id in self.actor_groups:
            return model_id

        master_addr = "127.0.0.1"
        master_port = str(get_free_port())
        num_gpus = 1  # num gpus per worker is 1 for tensor parallelism

        cleaned_actor_name = model_id.replace("/", "_").replace(":", "_").lower()
        workers = []
        for rank in range(world_size):
            workers.append(
                TrainingActor.options(
                    num_gpus=num_gpus,
                    get_if_exists=True,
                    lifetime="detached",
                    name=f"training_actor_{cleaned_actor_name}_{rank}",
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
                )
            )

        # Store for later querying
        self.actor_groups[model_id] = ActorGroup(
            workers=workers,
            model_id=model_id,
            status=ActorStatus.INITIALIZING,
            max_wait_time=self.max_wait_time,
        )

        await self.start()

        return model_id
