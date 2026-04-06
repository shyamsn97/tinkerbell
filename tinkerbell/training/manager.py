from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fastapi import HTTPException

from tinker.types import LoraConfig, LossFnType

from tinkerbell.store import GlobalStore
from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.optimizer import OptimStepRequest, ZeroGradRequest
from tinkerbell.types.requests import ForwardBackwardRequest, ForwardRequest
from tinkerbell.types.responses import RemoteFuture
from tinkerbell.utils import clean_model_name, get_actor_names_by_prefix, get_free_port

logger = logging.getLogger(__name__)


class ActorStatus(Enum):
    READY = "ready"
    PENDING = "pending"
    NOT_PRESENT = "not_present"


def _make_error_result(error: str | Exception, error_type: str | None = None) -> dict:
    """Create standardized error result dict."""
    err_str = str(error)
    return {
        "success": False,
        "error": err_str,
        "error_type": (
            error_type or type(error).__name__
            if isinstance(error, Exception)
            else "Error"
        ),
        "message": f"Error: {err_str}",
    }


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

    async def _broadcast(self, method: str, *args, **kwargs) -> list[Any]:
        """Broadcast a method call to all workers and gather results."""
        refs = [getattr(w, method).remote(*args, **kwargs) for w in self.workers]
        return await asyncio.gather(*refs)

    async def zero_grad(self) -> None:
        await self._broadcast("zero_grad")

    async def forward_backward(
        self,
        data: list[Any],
        loss_fns: list[LossFnType],
        adapter_name: Optional[str] = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
    ) -> list[dict[str, Any]]:
        await self._ensure_ready()
        outputs = await asyncio.gather(
            *[
                w.forward_backward.remote(
                    data=data,
                    loss_fns=loss_fns,
                    adapter_name=adapter_name,
                    forward_kwargs=forward_kwargs,
                    return_logprobs=return_logprobs,
                )
                for w in self.workers
            ]
        )
        return self._restructure_outputs(outputs, len(data))

    async def optim_step(
        self, adapter_name: Optional[str] = None, optimizer_params: dict[str, Any] = {}
    ) -> None:
        await self._broadcast(
            "optim_step", adapter_name=adapter_name, optimizer_params=optimizer_params
        )

    async def save_checkpoint(
        self, checkpoint_path: str, adapter_name: str | None = None
    ) -> None:
        await self._broadcast("save_checkpoint", checkpoint_path, adapter_name)
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
        await self._broadcast(
            "push_to_hub",
            repo_id=repo_id,
            adapter_name=adapter_name,
            token=token,
            private=private,
            commit_message=commit_message,
            push_kwargs=push_kwargs,
        )

    async def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> None:
        await self._broadcast("add_adapter", adapter_name, lora_config)

    async def set_active_adapter(self, adapter_name: str) -> None:
        await self._broadcast("set_active_adapter", adapter_name)

    def _restructure_outputs(
        self, outputs: list[dict[str, Any]], batch_size: int
    ) -> list[dict[str, Any]]:
        rank_0_output = next((o for o in outputs if o is not None), None)
        if rank_0_output is None:
            return []
        return [
            {
                k: (v[i] if isinstance(v, list) and i < len(v) else v)
                for k, v in rank_0_output.items()
            }
            for i in range(batch_size)
        ]

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
        cleaned_name = clean_model_name(model_name)
        try:
            all_actors = ray.util.list_named_actors(all_namespaces=True)
        except TypeError:
            all_actors = ray.util.list_named_actors()

        actors_list = [
            {"name": a, "namespace": "tinkerbell"} if isinstance(a, str) else a
            for a in all_actors
            if isinstance(a, (str, dict))
        ]
        actor_names = get_actor_names_by_prefix(
            f"training_actor_{cleaned_name}", actors_list
        )
        if not actor_names:
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
        initialize_base_model: bool = False,
        max_wait_time: float = 600.0,
    ) -> ActorGroup:
        master_addr, master_port = "127.0.0.1", str(get_free_port())
        cleaned_name = clean_model_name(model_name)
        workers = [
            TrainingActor.options(
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
                initialize_base_model=initialize_base_model,
            )
            for rank in range(world_size)
        ]
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
        clock_cycle: float = 2.0,
        global_store: GlobalStore = None,
    ):
        self.actor_groups: Dict[str, ActorGroup] = {}
        self.max_wait_time = max_wait_time
        self.clock_cycle = clock_cycle
        self.global_store = global_store
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.running = False
        self.immediate_trigger: asyncio.Event = asyncio.Event()

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
        immediate: bool = False,
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
        result = await self._queue_and_process(request, model_name)
        if immediate:
            self.immediate_trigger.set()
        return result

    async def zero_grad(
        self,
        model_name: str,
        adapter_name: Optional[str] = None,
        immediate: bool = False,
    ) -> RemoteFuture:
        from tinkerbell.types.optimizer import ZeroGradRequest

        request = ZeroGradRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            adapter_name=adapter_name,
        )
        await self.global_store.add_request_to_queue.remote(request=request)
        if immediate:
            self.immediate_trigger.set()
        return RemoteFuture(request_id=request.request_id, model_name=model_name)

    async def optim_step(
        self,
        model_name: str,
        adapter_name: Optional[str] = None,
        optimizer_params: dict[str, Any] = {},
        immediate: bool = False,
    ) -> RemoteFuture:
        request = OptimStepRequest(
            request_id=str(uuid.uuid4()),
            model_name=model_name,
            adapter_name=adapter_name,
            optimizer_params=optimizer_params,
        )
        await self.global_store.add_request_to_queue.remote(request=request)
        if immediate:
            self.immediate_trigger.set()
        return RemoteFuture(request_id=request.request_id, model_name=model_name)

    async def save_checkpoint(
        self, model_name: str, checkpoint_path: str, adapter_name: str | None = None
    ) -> None:
        await self._get_actor_group_or_raise(model_name).save_checkpoint(
            checkpoint_path, adapter_name
        )
        # Sync adapter paths to GlobalStore
        if adapter_name:
            try:
                await self.global_store.update_training_actor_adapters.remote(
                    model_name, adapter_name, checkpoint_path
                )
            except Exception:
                pass

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
        initialize_base_model: bool = False,
    ) -> str:
        await self.start()
        model_name = model_name or clean_model_name(base_model)
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

        # First, try to get from GlobalStore (fastest, shared across replicas)
        actor_group = await self._get_actor_group_from_global_store(
            model_name, base_model
        )
        if actor_group is not None:
            self.actor_groups[model_name] = actor_group
            if adapter_name and lora_config_dict:
                await actor_group.add_adapter(adapter_name, lora_config_dict)
            logger.info(f"Got training actors '{model_name}' from GlobalStore")
            return model_name

        # Fallback: try to reconnect to existing detached actors by name
        try:
            existing = ActorGroup.try_reconnect_to_existing_actors(
                model_name, base_model, self.max_wait_time
            )
            self.actor_groups[model_name] = existing
            # Store in GlobalStore for other replicas
            await self._store_actor_group_in_global_store(existing)
            if adapter_name and lora_config_dict:
                await existing.add_adapter(adapter_name, lora_config_dict)
            logger.info(f"Reconnected to existing training actors '{model_name}'")
            return model_name
        except Exception:
            pass

        actor_group = ActorGroup.create_actor_group(
            world_size=world_size,
            base_model=base_model,
            model_name=model_name,
            model_kwargs=model_kwargs,
            parallelize_plan=parallelize_plan,
            scheduler_params=scheduler_params,
            ray_worker_options=ray_worker_options,
            lora_config=lora_config_dict,
            adapter_name=adapter_name,
            initialize_base_model=initialize_base_model,
            max_wait_time=self.max_wait_time,
        )
        self.actor_groups[model_name] = actor_group
        # Store in GlobalStore for cross-replica access
        await self._store_actor_group_in_global_store(actor_group)
        return model_name

    def _get_actor_group_or_raise(
        self, model_name: str, base_model: str = ""
    ) -> ActorGroup:
        if model_name in self.actor_groups:
            return self.actor_groups[model_name]

        # Try to get from GlobalStore first (sync version for non-async callers)
        try:
            actor_info = ray.get(
                self.global_store.get_training_actors.remote(model_name), timeout=5.0
            )
            if actor_info is not None:
                workers = actor_info["workers"]
                # Verify at least one worker is alive by calling setup (idempotent)
                try:
                    ray.get(workers[0].setup.remote(), timeout=5.0)
                    actor_group = ActorGroup(
                        workers=workers,
                        base_model=actor_info["base_model"],
                        model_name=model_name,
                        status=ActorStatus.READY,
                        max_wait_time=self.max_wait_time,
                    )
                    actor_group.adapters = actor_info.get("adapters", {})
                    self.actor_groups[model_name] = actor_group
                    logger.info(f"Got training actors '{model_name}' from GlobalStore")
                    return actor_group
                except Exception as e:
                    logger.warning(
                        f"Training actors '{model_name}' from GlobalStore are unresponsive: {e}"
                    )
        except Exception as e:
            logger.debug(f"Could not get training actors from GlobalStore: {e}")

        # Fallback: try to reconnect to existing detached actors
        existing = ActorGroup.try_reconnect_to_existing_actors(
            model_name, base_model or model_name, self.max_wait_time
        )
        self.actor_groups[model_name] = existing
        # Store in GlobalStore for other replicas
        try:
            ray.get(
                self.global_store.set_training_actors.remote(
                    model_name=model_name,
                    workers=existing.workers,
                    base_model=existing.base_model,
                    adapters=existing.adapters,
                )
            )
        except Exception:
            pass
        return existing

    async def _queue_and_process(
        self, request: ForwardRequest | ForwardBackwardRequest, model_name: str
    ) -> RemoteFuture:
        await self.global_store.add_request_to_queue.remote(request=request)
        return RemoteFuture(request_id=request.request_id, model_name=model_name)

    async def _store_actor_group_in_global_store(self, actor_group: ActorGroup) -> None:
        """Store actor group workers in GlobalStore for cross-replica access."""
        try:
            await self.global_store.set_training_actors.remote(
                model_name=actor_group.model_name,
                workers=actor_group.workers,
                base_model=actor_group.base_model,
                adapters=actor_group.adapters,
            )
        except Exception as e:
            logger.warning(f"Failed to store actor group in GlobalStore: {e}")

    async def _get_actor_group_from_global_store(
        self, model_name: str, base_model: str
    ) -> Optional[ActorGroup]:
        """Try to get actor group from GlobalStore."""
        try:
            actor_info = await self.global_store.get_training_actors.remote(model_name)
            if actor_info is None:
                return None

            workers = actor_info["workers"]
            # Verify at least one worker is alive by calling setup (idempotent)
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(ray.get, workers[0].setup.remote()),
                    timeout=5.0,
                )
            except Exception as e:
                logger.warning(
                    f"Training actors '{model_name}' from GlobalStore are unresponsive: {e}"
                )
                return None

            actor_group = ActorGroup(
                workers=workers,
                base_model=actor_info["base_model"],
                model_name=model_name,
                status=ActorStatus.READY,
                max_wait_time=self.max_wait_time,
            )
            actor_group.adapters = actor_info.get("adapters", {})
            return actor_group
        except Exception as e:
            logger.debug(f"Could not get actor group from GlobalStore: {e}")
            return None

    async def shutdown_training_actors(self, model_name: str) -> bool:
        """Shutdown training actors and clean up from registries."""
        if model_name not in self.actor_groups:
            return False

        actor_group = self.actor_groups[model_name]
        # Kill all workers
        for worker in actor_group.workers:
            try:
                ray.kill(worker)
            except Exception as e:
                logger.warning(f"Error killing training worker: {e}")

        # Remove from local cache
        del self.actor_groups[model_name]

        # Remove from GlobalStore
        try:
            await self.global_store.delete_training_actors.remote(model_name)
        except Exception:
            pass

        logger.info(f"Shut down training actors for '{model_name}'")
        return True

    async def _batch_processor_loop(self) -> None:
        while self.running:
            try:
                # Wait for either the clock cycle timeout or immediate trigger
                try:
                    await asyncio.wait_for(
                        self.immediate_trigger.wait(), timeout=self.clock_cycle
                    )
                    self.immediate_trigger.clear()
                except asyncio.TimeoutError:
                    pass  # Normal clock cycle timeout
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

    async def _process_forward_backward_batch(
        self,
        fb_requests: list[ForwardBackwardRequest],
        actor_group: ActorGroup,
        adapter_name: str | None,
    ) -> None:
        """Process a batch of forward_backward requests."""
        if not fb_requests:
            return

        batch_data, request_sizes, loss_fns = [], [], []
        for req in fb_requests:
            request_sizes.append(len(req.data))
            batch_data.extend(req.data)
            loss_fns.extend([req.loss_fn] * len(req.data))

        output = await actor_group.forward_backward(
            data=batch_data,
            adapter_name=adapter_name,
            forward_kwargs=fb_requests[0].forward_kwargs,
            return_logprobs=getattr(fb_requests[0], "return_logprobs", False),
            loss_fns=loss_fns,
        )

        output_idx = 0
        for req, req_size in zip(fb_requests, request_sizes):
            req_outputs = output[output_idx : output_idx + req_size]
            output_idx += req_size
            if req_outputs and len(req_outputs) > 0:
                first_output = req_outputs[0]
                if isinstance(first_output, dict) and first_output:
                    combined = {}
                    for key in first_output.keys():
                        if key == "sum_gradient":
                            combined[key] = first_output[key]
                        else:
                            combined[key] = [
                                o[key]
                                for o in req_outputs
                                if isinstance(o, dict) and key in o
                            ]
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=combined
                    )
                else:
                    error_result = _make_error_result(
                        "Empty or invalid output from forward_backward",
                        "InvalidOutputError",
                    )
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=error_result
                    )

    async def _process_optim_step(
        self,
        optim_request: OptimStepRequest,
        actor_group: ActorGroup,
    ) -> None:
        """Process an optim_step request."""
        await actor_group.optim_step(
            adapter_name=optim_request.adapter_name,
            optimizer_params=optim_request.optimizer_params,
        )
        await self.global_store.set_result.remote(
            request_id=optim_request.request_id,
            result={
                "model_name": optim_request.model_name,
                "message": f"Optimizer stepped for {optim_request.model_name}",
            },
        )

    async def _process_zero_grad(
        self,
        zero_grad_request: ZeroGradRequest,
        actor_group: ActorGroup,
    ) -> None:
        """Process a zero_grad request."""
        await actor_group.zero_grad()
        await self.global_store.set_result.remote(
            request_id=zero_grad_request.request_id,
            result={
                "model_name": zero_grad_request.model_name,
                "message": f"Gradients zeroed for {zero_grad_request.model_name}",
            },
        )

    async def _process_batch(self, queue_key: str) -> None:
        """Process a batch of requests, treating optim_step as barriers.

        Requests are processed in order:
        - ForwardBackwardRequests are accumulated and batched together
        - When an OptimStepRequest is encountered, flush the pending batch first,
          then execute the optim_step
        """
        requests = []
        try:
            model_name, adapter_name = self._parse_queue_key(queue_key)
            request_queue = await self.global_store.get_request_queue.remote()
            requests = request_queue.get(queue_key, [])
            if not requests:
                return

            await self.global_store.clear_request_queue.remote(queue_key=queue_key)
            if await self.get_actor_status(model_name) != ActorStatus.READY:
                error_result = _make_error_result(
                    f"Actors for {model_name} are not ready", "ActorNotReadyError"
                )
                for req in requests:
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=error_result
                    )
                return

            actor_group = self._get_actor_group_or_raise(model_name)
            pending_fb_requests: list[ForwardBackwardRequest] = []

            for req in requests:
                if isinstance(req, ZeroGradRequest):
                    # Flush pending forward_backward requests before zero_grad
                    await self._process_forward_backward_batch(
                        pending_fb_requests, actor_group, adapter_name
                    )
                    pending_fb_requests = []
                    # Execute zero_grad
                    await self._process_zero_grad(req, actor_group)
                elif isinstance(req, OptimStepRequest):
                    # Flush pending forward_backward requests before optim_step
                    await self._process_forward_backward_batch(
                        pending_fb_requests, actor_group, adapter_name
                    )
                    pending_fb_requests = []
                    # Execute optim_step
                    await self._process_optim_step(req, actor_group)
                elif isinstance(req, ForwardBackwardRequest):
                    pending_fb_requests.append(req)

            # Process any remaining forward_backward requests
            await self._process_forward_backward_batch(
                pending_fb_requests, actor_group, adapter_name
            )

        except Exception as e:
            logger.error(f"Batch processing error for {queue_key}: {e}", exc_info=True)
            error_result = _make_error_result(e)
            for req in requests:
                try:
                    await self.global_store.set_result.remote(
                        request_id=req.request_id, result=error_result
                    )
                except Exception as set_error:
                    logger.error(
                        f"Failed to set error result for request {req.request_id}: {set_error}"
                    )
            raise
