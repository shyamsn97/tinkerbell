"""Ray-side wrapper around `Trainer`.

One `TrainingActor` per TP rank. Responsibilities:
  - Bring up torch.distributed.
  - Own a `Trainer` instance.
  - Expose methods that the local `TrainGroup` proxy calls directly via
    `actor.method.remote(...)`.

Blocking I/O (HF upload, checkpoint save on large models) is offloaded to a
worker thread via `asyncio.to_thread` so this actor's event loop stays
responsive for other methods (shutdown, diagnostics).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

import ray
import torch
import torch.distributed as dist
from tinker.types import Datum, LoraConfig, LossFnType, TensorData

from tinkerbell.training.trainer import Trainer

logger = logging.getLogger(__name__)

DEFAULT_PUSH_TO_HUB_TIMEOUT_S = 1800.0


@ray.remote
class TrainingActor:
    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        base_model: str,
        model_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        parallelize_plan: dict[str, str] | None = None,
        lora_config: Optional[LoraConfig | dict[str, Any]] = None,
        adapter_name: Optional[str] = None,
        initialize_base_model: bool = False,
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.trainer = Trainer(
            rank=rank,
            world_size=world_size,
            base_model=base_model,
            model_kwargs=model_kwargs or {},
            parallelize_plan=parallelize_plan or {},
            lora_config=lora_config,
            adapter_name=adapter_name,
            initialize_base_model=initialize_base_model,
        )
        self.model_name = model_name or base_model

    def setup_distributed(self) -> None:
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"TrainingActor(rank={self.rank}) started but torch.cuda.is_available()=False. "
                f"Check Ray was launched with GPUs and num_gpus=1 is honored."
            )
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"TrainingActor(rank={self.rank}/{self.world_size}) bound to CUDA "
            f"device 0: {props.name} ({props.total_memory / 1e9:.1f} GB, "
            f"cc={props.major}.{props.minor}); "
            f"visible GPUs={torch.cuda.device_count()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )

    async def setup(self) -> bool:
        if self.trainer.ready:
            return True
        self.setup_distributed()
        self.trainer.setup_model()
        return True

    async def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> bool:
        self.trainer.add_adapter(adapter_name, lora_config)
        return True

    async def set_active_adapter(self, adapter_name: str) -> bool:
        self.trainer.set_active_adapter(adapter_name)
        return True

    async def get_adapters(self) -> list[str]:
        return self.trainer.adapters()

    async def forward(
        self,
        data: list[Datum],
        forward_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        logprobs = self.trainer.forward(data, forward_kwargs=forward_kwargs)
        if self.rank != 0:
            return None
        return {
            "model_name": self.model_name,
            "logprobs": TensorData.from_torch(logprobs.detach().cpu()),
        }

    async def forward_backward(
        self,
        data: list[Datum],
        adapter_name: str | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        loss_fns: list[LossFnType] | None = None,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> dict[str, Any] | None:
        return self.trainer.forward_backward(
            data=data,
            adapter_name=adapter_name,
            forward_kwargs=forward_kwargs,
            loss_fns=loss_fns,
            loss_fn_config=loss_fn_config,
        )

    async def zero_grad(self) -> None:
        self.trainer.zero_grad()

    async def optim_step(
        self,
        adapter_name: str | None = None,
        optimizer_params: dict[str, Any] | None = None,
    ) -> None:
        self.trainer.optim_step(
            adapter_name=adapter_name, optimizer_params=optimizer_params
        )

    async def save_checkpoint(
        self, checkpoint_path: str, adapter_name: str | None = None
    ) -> bool:
        # Save can be mildly blocking on large models; keep it on the event
        # loop since dist.barrier() must run synchronously across ranks.
        self.trainer.save_model(checkpoint_path, adapter_name=adapter_name)
        return self.rank == 0

    async def push_to_hub(
        self,
        repo_id: str,
        adapter_name: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
        push_kwargs: dict[str, Any] | None = None,
        timeout_s: float = DEFAULT_PUSH_TO_HUB_TIMEOUT_S,
    ) -> bool:
        """Push to HF Hub. Off-loads the blocking upload to a worker thread."""
        if self.rank == 0:

            def _do_push() -> None:
                self.trainer.push_to_hub(
                    repo_id=repo_id,
                    adapter_name=adapter_name,
                    token=token,
                    private=private,
                    commit_message=commit_message,
                    push_kwargs=push_kwargs,
                )

            try:
                await asyncio.wait_for(asyncio.to_thread(_do_push), timeout=timeout_s)
            except asyncio.TimeoutError as e:
                raise RuntimeError(
                    f"push_to_hub exceeded timeout of {timeout_s}s for repo={repo_id}. "
                    f"HF upload appears stuck; aborting so training can continue."
                ) from e

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return self.rank == 0

    async def cleanup(self) -> bool:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return self.rank == 0


TrainingWorker = TrainingActor
