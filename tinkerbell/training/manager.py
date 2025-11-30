from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fastapi import HTTPException

from tinkerbell.store import GlobalStore
from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.lora_config import LoraConfig
from tinkerbell.types.requests import (
    ForwardBackwardRequest,
    ForwardRequest,
)
from tinkerbell.types.responses import RemoteFuture
from tinkerbell.utils import get_actor_names_by_prefix, get_free_port

logger = logging.getLogger(__name__)


class ActorStatus(Enum):
    READY = "ready"
    PENDING = "pending"
    NOT_PRESENT = "not_present"


class ActorGroup:
    def __init__(
        self,
        workers: list[Any],
        model_id: str,
        model_name: str,
        status: ActorStatus = ActorStatus.PENDING,
        max_wait_time: float = 600.0,
    ):
        self.workers = workers
        self.setup_refs = [worker.setup.remote() for worker in self.workers]
        self.model_id = model_id
        self.model_name = model_name
        self.adapters: dict[str, str] = {}  # adapter_name -> checkpoint_path
        self.status = status
        self.max_wait_time = max_wait_time

    async def get_status(self) -> ActorStatus:
        if self.status == ActorStatus.PENDING:
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

    def _restructure_outputs(
        self, outputs: list[dict[str, Any]], batch_size: int
    ) -> list[dict[str, Any]]:
        """Restructure outputs from workers to per-sample format."""
        rank_0_output = [output for output in outputs if output is not None][0]
        batched_output = [{} for _ in range(batch_size)]
        for i in range(batch_size):
            for key in rank_0_output:
                batched_output[i][key] = rank_0_output[key][i]
        return batched_output

    async def forward_backward(
        self,
        data: list[Any],
        adapter_name: Optional[str] = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        """Forward and backward pass. Switches to adapter_name if provided."""
        await self._ensure_ready()
        outputs = await self._execute_forward_backward(
            data, adapter_name, forward_kwargs, return_logprobs
        )
        return self._restructure_outputs(outputs, len(data))

    async def optim_step(
        self, adapter_name: Optional[str] = None, optimizer_params: dict[str, Any] = {}
    ) -> None:
        """Step the optimizer for all workers."""
        refs = [
            worker.optim_step.remote(
                adapter_name=adapter_name, optimizer_params=optimizer_params
            )
            for worker in self.workers
        ]
        await asyncio.gather(*refs)

    async def save_checkpoint(
        self, checkpoint_path: str, adapter_name: str | None = None
    ) -> None:
        """Save checkpoint for all workers, optionally for a specific adapter."""
        refs = [
            worker.save_checkpoint.remote(checkpoint_path, adapter_name)
            for worker in self.workers
        ]
        await asyncio.gather(*refs)
        if adapter_name:
            self.adapters[adapter_name] = checkpoint_path

    async def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> None:
        """Add a new LoRA adapter to all workers."""
        refs = [
            worker.add_adapter.remote(adapter_name, lora_config)
            for worker in self.workers
        ]
        await asyncio.gather(*refs)

    async def set_active_adapter(self, adapter_name: str) -> None:
        """Set active adapter on all workers."""
        refs = [
            worker.set_active_adapter.remote(adapter_name) for worker in self.workers
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
        adapter_name: Optional[str],
        forward_kwargs: dict[str, Any],
        return_logprobs: bool,
    ) -> list[dict[str, Any]]:
        """Execute forward-backward pass on all workers."""
        refs = [
            worker.forward_backward.remote(
                data=data,
                adapter_name=adapter_name,
                forward_kwargs=forward_kwargs,
                return_logprobs=return_logprobs,
            )
            for worker in self.workers
        ]
        return await asyncio.gather(*refs)

    @classmethod
    def try_reconnect_to_existing_actors(
        cls, model_name: str, model_id: str, max_wait_time: float = 600.0
    ) -> ActorGroup:
        """Reconnect to existing actors by model_name."""
        cleaned_name = model_name.replace("/", "_").replace(":", "_").lower()
        actor_names = get_actor_names_by_prefix(
            f"training_actor_{cleaned_name}",
            ray.util.list_named_actors(namespace="tinkerbell"),
        )
        workers = [
            ray.get_actor(name=name, namespace="tinkerbell") for name in actor_names
        ]
        return cls(
            workers=workers,
            model_id=model_id,
            model_name=model_name,
            status=ActorStatus.PENDING,
            max_wait_time=max_wait_time,
        )

    @classmethod
    def create_actor_group(
        cls,
        world_size: int,
        model_id: str,
        model_name: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        scheduler_params: dict[str, Any],
        ray_worker_options: dict[str, Any] = {},
        lora_config: Optional[dict[str, Any]] = None,
        adapter_name: Optional[str] = None,
        initialize_random_weights: bool = False,
        max_wait_time: float = 600.0,
    ) -> ActorGroup:
        """Create training actors for the group."""
        master_addr = "127.0.0.1"
        master_port = str(get_free_port())
        cleaned_name = model_name.replace("/", "_").replace(":", "_").lower()

        workers = []
        for rank in range(world_size):
            worker = TrainingActor.options(
                num_gpus=1,
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
                adapter_name=adapter_name,
                initialize_random_weights=initialize_random_weights,
            )
            workers.append(worker)

        return cls(
            workers=workers,
            model_id=model_id,
            model_name=model_name,
            status=ActorStatus.PENDING,
            max_wait_time=max_wait_time,
        )


class TrainingManager:
    def __init__(
        self,
        max_wait_time: float = 600.0,
        clock_cycle: float = 5.0,
        global_store: GlobalStore = None,
    ):
        self.actor_groups: Dict[str, ActorGroup] = {}
        self.max_wait_time = max_wait_time
        self.clock_cycle = clock_cycle
        self.global_store = global_store
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

    async def get_actor_status(self, model_name: str) -> ActorStatus:
        """Check if training actors are ready."""
        if model_name not in self.actor_groups:
            return ActorStatus.NOT_PRESENT
        return await self.actor_groups[model_name].get_status()

    async def forward(
        self,
        model_name: str,
        data: list[Any],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """Queue a forward request."""
        request = ForwardRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_name,  # Use model_name as the routing key
            data=data,
            forward_kwargs=forward_kwargs,
        )
        return await self._queue_and_process(request, model_name)

    async def forward_backward(
        self,
        model_name: str,
        data: list[Any],
        adapter_name: Optional[str] = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> RemoteFuture:
        """Queue a forward-backward request."""
        request = ForwardBackwardRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_name,
            adapter_name=adapter_name,
            data=data,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
        )
        return await self._queue_and_process(request, model_name)

    async def zero_grad(self, model_name: str) -> None:
        """Zero gradients for all workers."""
        await self._get_actor_group_or_raise(model_name).zero_grad()

    async def optim_step(
        self,
        model_name: str,
        adapter_name: Optional[str] = None,
        optimizer_params: dict[str, Any] = {},
    ) -> None:
        """Step optimizer for all workers."""
        await self._get_actor_group_or_raise(model_name).optim_step(
            adapter_name=adapter_name, optimizer_params=optimizer_params
        )

    async def save_checkpoint(
        self, model_name: str, checkpoint_path: str, adapter_name: str | None = None
    ) -> None:
        """Save checkpoint, optionally for a specific adapter."""
        await self._get_actor_group_or_raise(model_name).save_checkpoint(
            checkpoint_path, adapter_name
        )

    async def set_active_adapter(self, model_name: str, adapter_name: str) -> None:
        """Set active adapter for training."""
        await self._get_actor_group_or_raise(model_name).set_active_adapter(
            adapter_name
        )

    def get_adapter_paths(self, model_name: str) -> dict[str, str]:
        """Get saved adapter paths for a model group."""
        if model_name in self.actor_groups:
            return self.actor_groups[model_name].adapters.copy()
        return {}

    async def create_training_actors(
        self,
        world_size: int,
        model_id: str,
        model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        model_kwargs: dict[str, Any] = {},
        parallelize_plan: dict[str, str] = {},
        scheduler_params: dict[str, Any] = {},
        ray_worker_options: dict[str, Any] = {},
        lora_config: Optional[LoraConfig | dict[str, Any]] = None,
        initialize_random_weights: bool = False,
    ) -> str:
        """Create or add adapter to training actors."""
        await self.start()
        model_name = model_name or model_id

        lora_config_dict = None
        if lora_config is not None:
            lora_config_dict = (
                lora_config
                if isinstance(lora_config, dict)
                else lora_config.model_dump()
            )

        # If actor group exists, add adapter to it
        if model_name in self.actor_groups:
            if adapter_name and lora_config_dict:
                await self.actor_groups[model_name].add_adapter(
                    adapter_name, lora_config_dict
                )
            return model_name

        # Try reconnect
        try:
            existing = ActorGroup.try_reconnect_to_existing_actors(
                model_name, model_id, self.max_wait_time
            )
            self.actor_groups[model_name] = existing
            if adapter_name and lora_config_dict:
                await existing.add_adapter(adapter_name, lora_config_dict)
            return model_name
        except Exception:
            pass

        # Create new actor group
        self.actor_groups[model_name] = ActorGroup.create_actor_group(
            world_size=world_size,
            model_id=model_id,
            model_name=model_name,
            model_kwargs=model_kwargs,
            parallelize_plan=parallelize_plan,
            scheduler_params=scheduler_params,
            ray_worker_options=ray_worker_options,
            lora_config=lora_config_dict,
            adapter_name=adapter_name,
            initialize_random_weights=initialize_random_weights,
            max_wait_time=self.max_wait_time,
        )
        return model_name

    def _get_actor_group_or_raise(
        self, model_name: str, model_id: str = ""
    ) -> ActorGroup:
        """Get actor group or raise ValueError if not found."""
        if model_name not in self.actor_groups:
            existing = ActorGroup.try_reconnect_to_existing_actors(
                model_name, model_id or model_name, self.max_wait_time
            )
            self.actor_groups[model_name] = existing
        return self.actor_groups[model_name]

    async def _queue_and_process(
        self, request: ForwardRequest | ForwardBackwardRequest, model_name: str
    ) -> RemoteFuture:
        """Queue a request and optionally process immediately."""
        await self.global_store.add_request_to_queue.remote(request=request)
        if self.clock_cycle <= 0.0:
            await self._process_batch(model_name)
        return RemoteFuture(request_id=request.request_id, model_id=model_name)

    async def _batch_processor_loop(self) -> None:
        """Background task that processes batches at regular intervals."""
        logger.info("[_batch_processor_loop] Batch processor loop started")
        while self.running:
            try:
                await asyncio.sleep(self.clock_cycle)
                logger.info(
                    "[_batch_processor_loop] Woke up, processing all batches..."
                )
                await self._process_all_batches()
            except asyncio.CancelledError:
                logger.info("[_batch_processor_loop] Batch processor cancelled")
                break
            except Exception as e:
                logger.error(
                    f"[_batch_processor_loop] Error in batch processor: {e}",
                    exc_info=True,
                )
                await self.stop()
                raise e

    async def _process_all_batches(self) -> None:
        """Process all pending batches across all model/adapter groups."""
        request_queue = await self.global_store.get_request_queue.remote()
        # Queue keys are "model_name:adapter_name" format
        for queue_key in list(request_queue.keys()):
            if len(request_queue[queue_key]) > 0:
                await self._process_batch(queue_key)

    def _parse_queue_key(self, queue_key: str) -> tuple[str, str | None]:
        """Parse queue key into (model_name, adapter_name)."""
        parts = queue_key.split(":", 1)
        model_name = parts[0]
        adapter_name = parts[1] if len(parts) > 1 and parts[1] else None
        return model_name, adapter_name

    def _prepare_batch_data(self, requests: list[Any]) -> tuple[list[Any], list[int]]:
        """Flatten request data into a single batch and track sizes."""
        batch_data = []
        request_sizes = []
        for req in requests:
            request_sizes.append(len(req.data))
            batch_data.extend(req.data)
        return batch_data, request_sizes

    async def _process_batch(self, queue_key: str) -> None:
        """Process a batch of requests for a model/adapter combination."""
        try:
            model_name, adapter_name = self._parse_queue_key(queue_key)
            logger.info(
                f"[_process_batch] Processing batch for: {queue_key} (model={model_name}, adapter={adapter_name})"
            )
            requests = await self._get_pending_requests(queue_key)
            logger.info(f"[_process_batch] Found {len(requests)} pending requests")
            if not requests:
                return

            await self.global_store.clear_request_queue.remote(queue_key=queue_key)

            status = await self.get_actor_status(model_name)
            if status != ActorStatus.READY:
                logger.warning(f"[_process_batch] Actors not ready for {model_name}")
                return

            batch_data, request_sizes = self._prepare_batch_data(requests)
            output = await self._execute_batch(
                model_name, adapter_name, batch_data, requests[0]
            )
            await self._distribute_results(requests, output, request_sizes)
            logger.info(f"[_process_batch] Batch processing complete")
        except Exception as e:
            logger.error(
                f"[_process_batch] EXCEPTION for {queue_key}: {e}", exc_info=True
            )
            raise

    async def _get_pending_requests(self, queue_key: str) -> list[Any]:
        """Get all pending requests for a queue key."""
        request_queue = await self.global_store.get_request_queue.remote()
        return request_queue.get(queue_key, [])

    async def _execute_batch(
        self,
        model_name: str,
        adapter_name: str | None,
        batch_data: list[Any],
        sample_request: Any,
    ) -> list[dict[str, Any]]:
        """Execute the batched forward-backward pass."""
        actor_group = self._get_actor_group_or_raise(model_name)
        return await actor_group.forward_backward(
            data=batch_data,
            adapter_name=adapter_name,
            forward_kwargs=sample_request.forward_kwargs,
            return_logprobs=getattr(sample_request, "return_logprobs", False),
        )

    def _combine_request_outputs(
        self,
        req_outputs: list[dict[str, Any]],
    ) -> dict[str, list[Any]]:
        """Combine outputs for a single request into a dict of lists."""
        combined_output = {}
        for key in req_outputs[0].keys():
            combined_output[key] = [out[key] for out in req_outputs]
        return combined_output

    async def _distribute_results(
        self,
        requests: list[Any],
        output: list[dict[str, Any]],
        request_sizes: list[int],
    ) -> None:
        """Distribute batch results back to individual requests."""
        try:
            logger.info(
                f"[_distribute_results] Distributing results for {len(requests)} requests"
            )
            logger.info(f"[_distribute_results] Output: {output}")
            output_idx = 0
            for req, req_size in zip(requests, request_sizes):
                logger.info(
                    f"[_distribute_results] Processing request {req.request_id}, req_size: {req_size}"
                )
                req_outputs = output[output_idx : output_idx + req_size]
                output_idx += req_size

                if req_outputs:
                    logger.info(f"[_distribute_results] req_outputs: {req_outputs}")
                    combined_output = self._combine_request_outputs(req_outputs)
                    logger.info(
                        f"[_distribute_results] combined_output: {combined_output}"
                    )
                    logger.info(
                        f"[_distribute_results] Storing result for request_id: {req.request_id}"
                    )
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=combined_output
                    )
                    logger.info(
                        f"[_distribute_results] Successfully stored result for request_id: {req.request_id}"
                    )
                else:
                    logger.warning(
                        f"[_distribute_results] No outputs for request {req.request_id}"
                    )
        except Exception as e:
            logger.error(
                f"[_distribute_results] EXCEPTION distributing results: {e}",
                exc_info=True,
            )
            raise
