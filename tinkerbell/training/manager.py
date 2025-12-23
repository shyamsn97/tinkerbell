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
from tinkerbell.types.loss_fn_type import LossFnType
from tinkerbell.types.requests import ForwardBackwardRequest, ForwardRequest
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
        base_model: str,
        model_name: str,
        status: ActorStatus = ActorStatus.PENDING,
        max_wait_time: float = 600.0,
    ):
        self.workers = workers
        self.setup_refs = [worker.setup.remote() for worker in self.workers]
        self.base_model = base_model
        self.model_name = model_name
        self.adapters: dict[str, str] = {}
        self.status = status
        self.max_wait_time = max_wait_time

    async def get_status(self) -> ActorStatus:
        if self.status == ActorStatus.PENDING:
            await self._check_initialization_complete()
        return self.status

    async def wait_until_ready(self) -> bool:
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < self.max_wait_time:
            if await self.get_status() == ActorStatus.READY:
                return True
            await asyncio.sleep(1.0)
        return False

    async def zero_grad(self) -> None:
        refs = [worker.zero_grad.remote() for worker in self.workers]
        await asyncio.gather(*refs)

    async def forward_backward(
        self,
        data: list[Any],
        loss_fns: list[LossFnType],
        adapter_name: Optional[str] = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        await self._ensure_ready()
        refs = [
            worker.forward_backward.remote(
                data=data,
                loss_fns=loss_fns,
                adapter_name=adapter_name,
                forward_kwargs=forward_kwargs,
                return_logprobs=return_logprobs,
            )
            for worker in self.workers
        ]
        outputs = await asyncio.gather(*refs)
        return self._restructure_outputs(outputs, len(data))

    async def optim_step(
        self, adapter_name: Optional[str] = None, optimizer_params: dict[str, Any] = {}
    ) -> None:
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
        refs = [
            worker.save_checkpoint.remote(checkpoint_path, adapter_name)
            for worker in self.workers
        ]
        await asyncio.gather(*refs)
        if adapter_name:
            self.adapters[adapter_name] = checkpoint_path

    async def push_to_hub(
        self,
        repo_id: str,
        adapter_name: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
        push_kwargs: dict[str, Any] = {},
    ) -> None:
        refs = [
            worker.push_to_hub.remote(
                repo_id=repo_id,
                adapter_name=adapter_name,
                token=token,
                private=private,
                commit_message=commit_message,
                push_kwargs=push_kwargs,
            )
            for worker in self.workers
        ]
        await asyncio.gather(*refs)

    async def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> None:
        refs = [
            worker.add_adapter.remote(adapter_name, lora_config)
            for worker in self.workers
        ]
        await asyncio.gather(*refs)

    async def set_active_adapter(self, adapter_name: str) -> None:
        refs = [
            worker.set_active_adapter.remote(adapter_name) for worker in self.workers
        ]
        await asyncio.gather(*refs)

    def _restructure_outputs(
        self, outputs: list[dict[str, Any]], batch_size: int
    ) -> list[dict[str, Any]]:
        # Find the first non-None output (should be from rank 0)
        rank_0_output = None
        for o in outputs:
            if o is not None:
                rank_0_output = o
                break

        if rank_0_output is None:
            # All outputs are None, return empty list
            return []

        # Restructure outputs: some values are lists (per-item), others are batch-level (same for all)
        result = []
        for i in range(batch_size):
            item_output = {}
            for key, value in rank_0_output.items():
                # If value is a list, index into it; otherwise use the same value for all items
                if isinstance(value, list):
                    item_output[key] = value[i] if i < len(value) else None
                else:
                    # For non-list values (like sum_gradient dict), use the same value for all items
                    item_output[key] = value
            result.append(item_output)
        return result

    async def _check_initialization_complete(self) -> None:
        ready, _ = ray.wait(
            self.setup_refs, num_returns=len(self.setup_refs), timeout=0
        )
        if len(ready) == len(self.setup_refs):
            await asyncio.gather(*self.setup_refs)
            self.status = ActorStatus.READY

    async def _ensure_ready(self) -> None:
        if self.status != ActorStatus.READY:
            if not await self.wait_until_ready():
                raise HTTPException(
                    status_code=503, detail=f"Actors for {self.base_model} not ready"
                )

    @classmethod
    def try_reconnect_to_existing_actors(
        cls, model_name: str, base_model: str, max_wait_time: float = 600.0
    ) -> ActorGroup:
        cleaned_name = model_name.replace("/", "_").replace(":", "_").lower()

        # Get all actors and filter by namespace
        # Handle different Ray API versions
        try:
            # Try newer API with all_namespaces
            all_actors = ray.util.list_named_actors(all_namespaces=True)
        except TypeError:
            # Fall back to older API (no namespace filtering)
            all_actors = ray.util.list_named_actors()

        # Normalize actor format - handle both string and dict formats
        actors_list = []
        for actor in all_actors:
            if isinstance(actor, str):
                # String format - assume it's in the tinkerbell namespace
                actors_list.append({"name": actor, "namespace": "tinkerbell"})
            elif isinstance(actor, dict):
                actors_list.append(actor)
            else:
                # Unknown format, skip
                continue

        actor_names = get_actor_names_by_prefix(
            f"training_actor_{cleaned_name}",
            actors_list,
        )

        if not actor_names:
            # No existing actors found, raise exception to trigger creation
            raise ValueError(f"No existing actors found for model {model_name}")

        workers = [
            ray.get_actor(name=name, namespace="tinkerbell") for name in actor_names
        ]
        return cls(
            workers=workers,
            base_model=base_model,
            model_name=model_name,
            status=ActorStatus.PENDING,
            max_wait_time=max_wait_time,
        )

    @classmethod
    def create_actor_group(
        cls,
        world_size: int,
        base_model: str,
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
        master_addr, master_port = "127.0.0.1", str(get_free_port())
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
                base_model=base_model,
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
            base_model=base_model,
            model_name=model_name,
            status=ActorStatus.PENDING,
            max_wait_time=max_wait_time,
        )


class TrainingManager:
    def __init__(
        self,
        max_wait_time: float = 600.0,
        clock_cycle: float = 0.0,
        global_store: GlobalStore = None,
    ):
        self.actor_groups: Dict[str, ActorGroup] = {}
        self.max_wait_time = max_wait_time
        self.clock_cycle = clock_cycle
        self.global_store = global_store
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        if self.running:
            return
        self.running = True
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())

    async def stop(self):
        self.running = False
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

    async def get_actor_status(self, model_name: str) -> ActorStatus:
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
        request = ForwardRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
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
        loss_fn: LossFnType = "cross_entropy",
    ) -> RemoteFuture:
        request = ForwardBackwardRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            adapter_name=adapter_name,
            data=data,
            forward_kwargs=forward_kwargs,
            return_logprobs=return_logprobs,
            loss_fn=loss_fn,
        )
        return await self._queue_and_process(request, model_name)

    async def zero_grad(self, model_name: str) -> None:
        await self._get_actor_group_or_raise(model_name).zero_grad()

    async def optim_step(
        self,
        model_name: str,
        adapter_name: Optional[str] = None,
        optimizer_params: dict[str, Any] = {},
    ) -> None:
        await self._get_actor_group_or_raise(model_name).optim_step(
            adapter_name=adapter_name, optimizer_params=optimizer_params
        )

    async def save_checkpoint(
        self, model_name: str, checkpoint_path: str, adapter_name: str | None = None
    ) -> None:
        await self._get_actor_group_or_raise(model_name).save_checkpoint(
            checkpoint_path, adapter_name
        )

    async def push_to_hub(
        self,
        model_name: str,
        repo_id: str,
        adapter_name: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
        push_kwargs: dict[str, Any] = {},
    ) -> None:
        await self._get_actor_group_or_raise(model_name).push_to_hub(
            repo_id=repo_id,
            adapter_name=adapter_name,
            token=token,
            private=private,
            commit_message=commit_message,
            push_kwargs=push_kwargs,
        )

    async def set_active_adapter(self, model_name: str, adapter_name: str) -> None:
        await self._get_actor_group_or_raise(model_name).set_active_adapter(
            adapter_name
        )

    def get_adapter_paths(self, model_name: str) -> dict[str, str]:
        if model_name in self.actor_groups:
            return self.actor_groups[model_name].adapters.copy()
        return {}

    async def create_training_actors(
        self,
        world_size: int,
        base_model: str,
        model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        model_kwargs: dict[str, Any] = {},
        parallelize_plan: dict[str, str] = {},
        scheduler_params: dict[str, Any] = {},
        ray_worker_options: dict[str, Any] = {},
        lora_config: Optional[LoraConfig | dict[str, Any]] = None,
        initialize_random_weights: bool = False,
    ) -> str:
        await self.start()
        model_name = (
            model_name or base_model.replace("/", "_").replace(":", "_").lower()
        )
        lora_config_dict = (
            lora_config
            if isinstance(lora_config, dict)
            else (lora_config.model_dump() if lora_config else None)
        )

        if model_name in self.actor_groups:
            if adapter_name and lora_config_dict:
                await self.actor_groups[model_name].add_adapter(
                    adapter_name, lora_config_dict
                )
            return model_name

        try:
            existing = ActorGroup.try_reconnect_to_existing_actors(
                model_name, base_model, self.max_wait_time
            )
            self.actor_groups[model_name] = existing
            if adapter_name and lora_config_dict:
                await existing.add_adapter(adapter_name, lora_config_dict)
            return model_name
        except Exception:
            pass

        self.actor_groups[model_name] = ActorGroup.create_actor_group(
            world_size=world_size,
            base_model=base_model,
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
        self, model_name: str, base_model: str = ""
    ) -> ActorGroup:
        if model_name not in self.actor_groups:
            existing = ActorGroup.try_reconnect_to_existing_actors(
                model_name, base_model or model_name, self.max_wait_time
            )
            self.actor_groups[model_name] = existing
        return self.actor_groups[model_name]

    async def _queue_and_process(
        self, request: ForwardRequest | ForwardBackwardRequest, model_name: str
    ) -> RemoteFuture:
        await self.global_store.add_request_to_queue.remote(request=request)
        return RemoteFuture(request_id=request.request_id, model_name=model_name)

    async def _batch_processor_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(self.clock_cycle)
                await self._process_all_batches()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await self.stop()
                raise

    async def _process_all_batches(self) -> None:
        request_queue = await self.global_store.get_request_queue.remote()
        for queue_key in list(request_queue.keys()):
            if request_queue[queue_key]:
                await self._process_batch(queue_key)

    def _parse_queue_key(self, queue_key: str) -> tuple[str, str | None]:
        parts = queue_key.split(":", 1)
        return parts[0], parts[1] if len(parts) > 1 and parts[1] else None

    async def _process_batch(self, queue_key: str) -> None:
        requests = []
        try:
            model_name, adapter_name = self._parse_queue_key(queue_key)
            request_queue = await self.global_store.get_request_queue.remote()
            requests = request_queue.get(queue_key, [])
            if not requests:
                return

            await self.global_store.clear_request_queue.remote(queue_key=queue_key)
            if await self.get_actor_status(model_name) != ActorStatus.READY:
                # Set error results for requests when actors aren't ready
                error_result = {
                    "success": False,
                    "error": f"Actors for {model_name} are not ready",
                    "error_type": "ActorNotReadyError",
                }
                for req in requests:
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=error_result
                    )
                return

            batch_data, request_sizes, loss_fns = [], [], []
            for req in requests:
                request_sizes.append(len(req.data))
                batch_data.extend(req.data)
                loss_fns.extend([req.loss_fn] * len(req.data))

            actor_group = self._get_actor_group_or_raise(model_name)
            output = await actor_group.forward_backward(
                data=batch_data,
                adapter_name=adapter_name,
                forward_kwargs=requests[0].forward_kwargs,
                return_logprobs=getattr(requests[0], "return_logprobs", False),
                loss_fns=loss_fns,
            )

            output_idx = 0
            for req, req_size in zip(requests, request_sizes):
                req_outputs = output[output_idx : output_idx + req_size]
                output_idx += req_size
                if req_outputs and len(req_outputs) > 0:
                    # Get keys from first output dict
                    first_output = req_outputs[0]
                    if isinstance(first_output, dict) and first_output:
                        combined = {}
                        for key in first_output.keys():
                            # sum_gradient is the same for all items in the batch, so use the first one
                            if key == "sum_gradient":
                                combined[key] = first_output[key]
                            else:
                                # For other keys, create a list (one value per item)
                                combined[key] = [
                                    o[key]
                                    for o in req_outputs
                                    if isinstance(o, dict) and key in o
                                ]
                        await self.global_store.set_result.remote(
                            request_id=req.request_id, result=combined
                        )
                    else:
                        # Empty or invalid output, set error result
                        error_result = {
                            "success": False,
                            "error": "Empty or invalid output from forward_backward",
                            "error_type": "InvalidOutputError",
                        }
                        await self.global_store.set_result.remote(
                            request_id=req.request_id, result=error_result
                        )
        except Exception as e:
            logger.error(f"Batch processing error for {queue_key}: {e}", exc_info=True)
            # Set error results for all requests so clients don't hang
            error_result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"Error processing batch: {str(e)}",
            }
            for req in requests:
                try:
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=error_result
                    )
                except Exception as set_error:
                    logger.error(
                        f"Failed to set error result for request {req.request_id}: {set_error}"
                    )
            # Re-raise to ensure it's logged at higher levels
            raise
