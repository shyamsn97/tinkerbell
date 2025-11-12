import asyncio
import time
import traceback
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fastapi import HTTPException

from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.requests import ForwardBackwardRequest, ForwardRequest
from tinkerbell.types.responses import RemoteFuture


class ActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"
    NOT_PRESENT = "not_present"


class ActorGroup:
    def __init__(
        self,
        workers: list[Any],
        model_name: str,
        status: ActorStatus = ActorStatus.INITIALIZING,
        max_wait_time: float = 600.0,
    ):
        self.workers = workers
        self.setup_refs = [worker.setup.remote() for worker in self.workers]
        self.model_name = model_name
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
                    detail=f"Actors for model {self.model_name} are not ready",
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
        print(f"Forward-backward refs: {refs}")
        outputs = await asyncio.gather(*refs)
        print(f"Forward-backward outputs: {outputs}")
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
        outputs = asyncio.gather(*refs)
        return [output for output in outputs if output is not None][0]

    async def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save the checkpoint for all workers."""
        refs = [
            worker.save_checkpoint.remote(checkpoint_path=checkpoint_path)
            for worker in self.workers
        ]
        outputs = asyncio.gather(*refs)
        return [output for output in outputs if output is not None][0]


@ray.remote
class GlobalStateManager:
    def __init__(self):
        self.request_queue: Dict[str, list[ForwardBackwardRequest]] = {}
        self.results: Dict[str, Any] = {}

    def get_request_queue(self) -> Dict[str, list[ForwardBackwardRequest]]:
        return self.request_queue

    def get_requests(self, model_name: str) -> list[ForwardBackwardRequest]:
        if model_name not in self.request_queue:
            return []
        return self.request_queue[model_name]

    def clear_request_queue(self, model_name: str):
        if model_name not in self.request_queue:
            return
        self.request_queue[model_name] = []

    def add_request_to_queue(self, request: ForwardBackwardRequest):
        if request.model_name not in self.request_queue:
            self.request_queue[request.model_name] = []
        self.request_queue[request.model_name].append(request)

    def get_results(self) -> Dict[str, Any]:
        return self.results

    def get_result(self, request_id: str) -> Any:
        if request_id not in self.results:
            return None
        return self.results[request_id]

    def set_result(self, request_id: str, result: Any):
        self.results[request_id] = result


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
        self.global_state = GlobalStateManager.options(
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
            results = ray.get(self.global_state.get_results.remote())
            if request_id in results:
                request_id_is_in_results = True
                break
            await asyncio.sleep(1.0)
            elapsed_time = time.time() - start_time
            print(f"Waiting for result {request_id} (elapsed: {elapsed_time:.1f}s)")
        if not request_id_is_in_results:
            raise TimeoutError(
                f"Result for request {request_id} not available after {max_wait_time} seconds"
            )
        return ray.get(self.global_state.get_result.remote(request_id=request_id))

    async def _batch_processor_loop(self) -> None:
        """Background task that processes batches at regular intervals."""
        while self.running:
            try:
                request_queue = ray.get(self.global_state.get_request_queue.remote())
                await asyncio.sleep(self.clock_cycle)
                print("Processing batches...")
                print(f"Request queue: {request_queue}")
                await self._process_all_batches()
            except asyncio.CancelledError:
                print("Batch processor task cancelled")
                break
            except Exception as e:
                # print(f"Error in batch processor: {e}")
                tb_str = traceback.format_exc()
                print(f"Error in batch processor: {e}\n{tb_str}")
                await self.stop()
                raise e

    async def _process_all_batches(self) -> None:
        """Process all pending batches across all model groups."""
        request_queue = ray.get(self.global_state.get_request_queue.remote())
        for model_name in list(request_queue.keys()):
            if len(request_queue[model_name]) > 0:
                await self._process_batch(model_name)

    async def _process_batch(self, model_name: str) -> None:
        """Process a batch of requests for a specific model."""
        # Get all pending requests
        request_queue = ray.get(self.global_state.get_request_queue.remote())
        requests = request_queue.get(model_name, [])
        print(f"Processing batch for model {model_name} with requests: {requests}")
        if len(requests) == 0:
            return

        self.global_state.clear_request_queue.remote(model_name=model_name)
        # Check if actor is ready
        try:
            status = await self.get_actor_status(model_name)
            print(f"Actor status: {status}")
            if status != ActorStatus.READY:
                return

            # Batch all inputs together
            batch_inputs = [req.inputs for req in requests]
            batch_targets = [req.targets for req in requests]
            # Assuming all requests use the same forward_kwargs (or merge them)
            forward_kwargs = requests[0].forward_kwargs if requests else {}
            return_logprobs = requests[0].return_logprobs if requests else False

            print(f"Batch inputs: {batch_inputs}")
            print(f"Forward kwargs: {forward_kwargs}")
            print(f"Batch targets: {batch_targets}")
            # Execute batch
            output = await self.actor_groups[model_name].forward_backward(
                inputs=batch_inputs,
                targets=batch_targets,
                forward_kwargs=forward_kwargs,
                return_logprobs=return_logprobs,
            )
            print(f"Output: {len(output)} outputs")
            print(f"Requests: {len(requests)} requests")
            # Distribute results back to futures

            for req, output in zip(requests, output, strict=True):
                if output is not None:
                    self.global_state.set_result.remote(
                        request_id=req.request_id, result=output
                    )
        except Exception as e:
            print(f"Error processing batch for model {model_name}: {e}")
            raise e

    async def get_actor_status(self, model_name: str) -> ActorStatus:
        """Check if training actors are ready."""
        if model_name not in self.actor_groups:
            return ActorStatus.NOT_PRESENT

        status = await self.actor_groups[model_name].get_status()
        return status

    async def forward(
        self,
        model_name: str,
        inputs: dict[str, Any],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """
        Queue a forward request to be processed in the next batch.
        Returns immediately with a future that will resolve when the batch is processed.
        """
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )

        request = ForwardRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )
        self.global_state.add_request_to_queue.remote(request=request)
        if self.clock_cycle <= 0.0:
            # act immediately
            await self._process_batch(model_name)
        return RemoteFuture(request_id=request.request_id)

    async def forward_backward(
        self,
        model_name: str,
        inputs: dict[str, Any],
        targets: Any = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """
        Queue a forward-backward request to be processed in the next batch.
        Returns immediately with a future that will resolve when the batch is processed.
        """
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )

        request = ForwardBackwardRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            inputs=inputs,
            targets=targets,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )

        # Add to queue
        self.global_state.add_request_to_queue.remote(request=request)

        if self.clock_cycle <= 0.0:
            # act immediately
            await self._process_batch(model_name)
        return RemoteFuture(request_id=request.request_id)

    async def zero_grad(self, model_name: str) -> None:
        """Zero the gradients for all workers."""
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )
        await self.actor_groups[model_name].zero_grad()

    async def optim_step(
        self, model_name: str, optimizer_params: dict[str, Any] = {}
    ) -> None:
        """Step the optimizer for all workers."""
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )
        await self.actor_groups[model_name].optim_step(
            optimizer_params=optimizer_params
        )

    async def save_checkpoint(self, model_name: str, checkpoint_path: str) -> None:
        """Save the checkpoint for all workers."""
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )
        await self.actor_groups[model_name].save_checkpoint(
            checkpoint_path=checkpoint_path
        )

    async def create_training_actors(
        self,
        world_size: int,
        master_addr: str,
        master_port: str,
        model_name: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        scheduler_params: dict[str, Any],
        ray_worker_options: dict[str, Any] = {},
    ) -> str:
        """Create a training worker for the given model name."""
        if model_name in self.actor_groups:
            print(f"Training actors for model {model_name} already exists...")
            return model_name

        num_gpus = 1  # num gpus per worker is 1 for tensor parallelism

        cleaned_actor_name = model_name.replace("/", "_").replace(":", "_").lower()
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
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    parallelize_plan=parallelize_plan,
                    scheduler_params=scheduler_params,
                )
            )

        # Store for later querying
        self.actor_groups[model_name] = ActorGroup(
            workers=workers,
            model_name=model_name,
            status=ActorStatus.INITIALIZING,
            max_wait_time=self.max_wait_time,
        )

        await self.start()

        return model_name
