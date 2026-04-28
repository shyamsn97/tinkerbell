"""TrainingEngine: coarse-grained handle over N TP-rank training workers.

Owns distribution, broadcast, and per-batch output restructuring for a
single (model, adapter) training job. Does not touch the WorkQueue or
JobStore — that's `TrainingExecutor`'s job (see `training/executor.py`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import ray
from tinker.types import LossFnType

from tinkerbell.state.registry import EngineRecord, GroupHandle
from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.route import RouteKey
from tinkerbell.utils import clean_model_name, get_free_port

logger = logging.getLogger(__name__)


class TrainingEngine:
    """Coarse-grained ops over N TP workers."""

    def __init__(self, handle: GroupHandle, max_wait_time: float = 600.0):
        self.handle = handle
        self.workers = handle.workers
        self.base_model = handle.base_model
        self.route = handle.route
        self.max_wait_time = max_wait_time
        # Fire setup.remote() once per worker so we can await readiness.
        self.setup_refs = [w.setup.remote() for w in self.workers]
        self._ready = False

    @property
    def adapters(self) -> dict[str, str]:
        return self.handle.adapters

    async def wait_until_ready(self) -> bool:
        if self._ready:
            return True
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < self.max_wait_time:
            ready, _ = ray.wait(
                self.setup_refs, num_returns=len(self.setup_refs), timeout=0
            )
            if len(ready) == len(self.setup_refs):
                await asyncio.gather(*self.setup_refs)
                self._ready = True
                return True
            await asyncio.sleep(1.0)
        return False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        ok = await self.wait_until_ready()
        if not ok:
            raise RuntimeError(
                f"Training engine for {self.route} not ready within {self.max_wait_time}s"
            )

    async def broadcast(self, method: str, *args, **kwargs) -> list[Any]:
        refs = [getattr(w, method).remote(*args, **kwargs) for w in self.workers]
        return await asyncio.gather(*refs)

    def restructure_outputs(
        self, outputs: list[Any], batch_size: int
    ) -> list[dict[str, Any]]:
        """Zip rank-0 output into per-example dicts."""
        rank_0 = next((o for o in outputs if o is not None), None)
        if rank_0 is None:
            return []
        return [
            {
                k: (v[i] if isinstance(v, list) and i < len(v) else v)
                for k, v in rank_0.items()
            }
            for i in range(batch_size)
        ]

    async def forward(
        self, data: list[Any], forward_kwargs: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        await self.ensure_ready()
        outputs = await self.broadcast(
            "forward", data=data, forward_kwargs=forward_kwargs or {}
        )
        return self.restructure_outputs(outputs, len(data))

    async def forward_backward(
        self,
        data: list[Any],
        loss_fns: list[LossFnType],
        adapter_name: Optional[str] = None,
        forward_kwargs: dict[str, Any] | None = None,
        loss_fn_config: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        await self.ensure_ready()
        outputs = await self.broadcast(
            "forward_backward",
            data=data,
            loss_fns=loss_fns,
            adapter_name=adapter_name,
            forward_kwargs=forward_kwargs or {},
            loss_fn_config=loss_fn_config,
        )
        return self.restructure_outputs(outputs, len(data))

    async def zero_grad(self) -> None:
        await self.broadcast("zero_grad")

    async def optim_step(
        self,
        adapter_name: Optional[str] = None,
        optimizer_params: dict[str, Any] | None = None,
    ) -> None:
        await self.broadcast(
            "optim_step",
            adapter_name=adapter_name,
            optimizer_params=optimizer_params or {},
        )

    async def save_checkpoint(
        self, checkpoint_path: str, adapter_name: str | None = None
    ) -> None:
        await self.broadcast("save_checkpoint", checkpoint_path, adapter_name)
        if adapter_name:
            self.handle.record.adapters[adapter_name] = checkpoint_path

    async def push_to_hub(self, **kwargs) -> None:
        await self.broadcast("push_to_hub", **kwargs)

    async def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> None:
        await self.broadcast("add_adapter", adapter_name, lora_config)

    async def set_active_adapter(self, adapter_name: str) -> None:
        await self.broadcast("set_active_adapter", adapter_name)


def spawn_training_engine(
    *,
    world_size: int,
    base_model: str,
    model_name: str,
    adapter_name: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    parallelize_plan: dict[str, str] | None = None,
    lora_config: dict[str, Any] | None = None,
    ray_worker_options: dict[str, Any] | None = None,
    initialize_base_model: bool = False,
    max_wait_time: float = 600.0,
) -> TrainingEngine:
    """Create N TP-rank TrainingActors as detached named Ray actors.

    Engines are keyed by `model_name` only (adapters live inside the engine).
    `adapter_name`, if given, is the first LoRA adapter installed on the actors.

    Callers must subsequently register the returned engine's record with the
    `Registry` and start a `TrainingExecutor` for it.
    """
    master_addr, master_port = "127.0.0.1", str(get_free_port())
    cleaned = clean_model_name(model_name)
    worker_names = [f"training_actor_{cleaned}_{rank}" for rank in range(world_size)]
    opts = ray_worker_options or {}

    workers = [
        TrainingActor.options(
            num_gpus=1,
            lifetime="detached",
            name=worker_names[rank],
            namespace="tinkerbell",
            **opts,
        ).remote(
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            base_model=base_model,
            model_kwargs=model_kwargs or {},
            parallelize_plan=parallelize_plan or {},
            lora_config=lora_config,
            adapter_name=adapter_name,
            initialize_base_model=initialize_base_model,
        )
        for rank in range(world_size)
    ]
    record = EngineRecord(
        kind="training",
        route=RouteKey(model=model_name, adapter=None),
        base_model=base_model,
        worker_names=worker_names,
    )
    return TrainingEngine(
        handle=GroupHandle(record=record, workers=workers),
        max_wait_time=max_wait_time,
    )


def attach_training_engine(
    handle: GroupHandle, max_wait_time: float = 600.0
) -> TrainingEngine:
    """Wrap an existing GroupHandle (resolved from Registry) as a TrainingEngine."""
    return TrainingEngine(handle=handle, max_wait_time=max_wait_time)
