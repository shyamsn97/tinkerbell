import asyncio
import traceback
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fastapi import HTTPException

from tinkerbell.types.requests import ForwardBackwardRequest, ForwardRequest
from tinkerbell.types.responses import RemoteFuture
from tinkerbell.training.actor import TrainingActor


class ActorStatus(Enum):
    READY = "ready"
    INITIALIZING = "initializing"


class ActorGroup:
    def __init__(
        self,
        workers: list[Any],
        setup_refs: list[Any],
        model_name: str,
        status: ActorStatus = ActorStatus.INITIALIZING,
    ):
        self.workers = workers
        self.setup_refs = setup_refs
        self.model_name = model_name
        self.status = status

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

    async def forward_backward(
        self,
        inputs: list[dict[str, Any]],
        targets: Any = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Forward and backward pass through the model with batched inputs."""
        if self.status != ActorStatus.READY:
            if self.status != ActorStatus.READY:
                raise HTTPException(
                    status_code=503,
                    detail=f"Actors for model {self.model_name} are not ready",
                )
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
        return outputs


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
        self.request_queue: Dict[str, list[ForwardBackwardRequest]] = {}
        self.results: Dict[str, Any] = {}
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

    async def get_result(self, request_id: str) -> Dict[str, Any]:
        """Get the result of a forward-backward request."""
        # Wait for the result to be available
        while request_id not in self.results:
            await asyncio.sleep(1.0)
            print(f"Waiting for result {request_id}")
        return self.results[request_id]

    async def _batch_processor_loop(self) -> None:
        """Background task that processes batches at regular intervals."""
        while self.running:
            try:
                await asyncio.sleep(self.clock_cycle)
                print("Processing batches...")
                print(f"Request queue: {self.request_queue}")
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
        for model_name in list(self.request_queue.keys()):
            if len(self.request_queue[model_name]) > 0:
                await self._process_batch(model_name)

    async def _process_batch(self, model_name: str) -> None:
        """Process a batch of requests for a specific model."""
        # Get all pending requests
        requests = self.request_queue.get(model_name, [])
        print(f"Processing batch for model {model_name} with requests: {requests}")
        if len(requests) == 0:
            return

        # Clear the queue for this model
        self.request_queue[model_name] = []

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
            # print(f"Output: {output}")
            # Distribute results back to futures
            for req, output in zip(requests, output, strict=True):
                if output is not None:
                    req.future.set_result(output)
                    self.results[req.request_id] = output
                else:
                    req.future.set_exception(
                        HTTPException(
                            status_code=500,
                            detail=f"Error processing request {req.request_id}",
                        )
                    )

        except Exception as e:
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)

    async def get_actor_status(self, model_name: str) -> ActorStatus:
        """Check if training actors are ready."""
        if model_name not in self.actor_groups:
            return ActorStatus.INITIALIZING

        status = await self.actor_groups[model_name].get_status()
        return status

    def _create_future(self, model_name: str) -> asyncio.Future:
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )

        # Initialize queue for this model if needed
        if model_name not in self.request_queue:
            self.request_queue[model_name] = []

        # Create a future for this request
        future = asyncio.get_event_loop().create_future()
        return future

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
        future = self._create_future(model_name)
        request = ForwardRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            inputs=inputs,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
            future=future,
        )
        self.request_queue[model_name].append(request)
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
        future = self._create_future(model_name)
        request = ForwardBackwardRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            inputs=inputs,
            targets=targets,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
            future=future,
        )

        # Add to queue
        self.request_queue[model_name].append(request)

        if self.clock_cycle <= 0.0:
            # act immediately
            await self._process_batch(model_name)
        return RemoteFuture(request_id=request.request_id)

    async def create_training_actors(
        self,
        world_size: int,
        master_addr: str,
        master_port: str,
        model_name: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        optimizer_params: dict[str, Any],
        scheduler_params: dict[str, Any],
        ray_worker_options: dict[str, Any] = {},
    ) -> str:
        """Create a training worker for the given model name."""
        if model_name in self.actor_groups:
            print(f"Training actors for model {model_name} already exists...")
            return model_name

        num_gpus = 1  # num gpus per worker is 1 for tensor parallelism

        ray_training_actor = ray.remote(TrainingActor).options(
            num_gpus=num_gpus, **ray_worker_options
        )
        workers = []
        for rank in range(world_size):
            workers.append(
                ray_training_actor.remote(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    parallelize_plan=parallelize_plan,
                    optimizer_params=optimizer_params,
                    scheduler_params=scheduler_params,
                )
            )

        # Trigger setup but don't wait
        setup_refs = [worker.setup.remote() for worker in workers]

        # Store for later querying
        self.actor_groups[model_name] = ActorGroup(
            workers=workers,
            setup_refs=setup_refs,
            model_name=model_name,
            status=ActorStatus.INITIALIZING,
        )

        self.request_queue[model_name] = []

        await self.start()

        return model_name
