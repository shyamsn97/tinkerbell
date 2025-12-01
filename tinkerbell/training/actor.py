import logging
import os
from typing import Any, Dict, Optional

import ray
import torch
import torch.distributed as dist

from tinkerbell.training.llm import LLM
from tinkerbell.training.loss import LOSSES
from tinkerbell.types.datum import Datum
from tinkerbell.types.lora_config import LoraConfig
from tinkerbell.types.loss_fn_type import LossFnType

logger = logging.getLogger(__name__)


@ray.remote
class TrainingActor:
    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        model_id: str,
        model_kwargs: dict[str, Any] = {},
        parallelize_plan: dict[str, str] = {},
        scheduler_params: dict[str, Any] = {},
        lora_config: Optional[LoraConfig | dict[str, Any]] = None,
        adapter_name: Optional[str] = None,
        initialize_random_weights: bool = False,
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
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.scheduler_params = scheduler_params
        self.initialize_random_weights = initialize_random_weights
        self.adapter_name = adapter_name
        self.training_model = None
        self.optimizer = None
        self.ready = False
        self.lora_config = (
            LoraConfig(**lora_config) if isinstance(lora_config, dict) else lora_config
        )

    def _get_optimizer(self, optimizer_config: dict[str, Any]) -> torch.optim.Optimizer:
        optimizer_name = optimizer_config.pop("name", "adam").lower()
        optimizer_dict = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
            "adadelta": torch.optim.Adadelta,
        }
        optimizer_config["foreach"] = False
        trainable_params = [
            p for p in self.training_model.model.parameters() if p.requires_grad
        ]
        return optimizer_dict[optimizer_name](trainable_params, **optimizer_config)

    def _setup_distributed(self):
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(0)

    def setup(self):
        if self.ready:
            return True
        self._setup_distributed()
        self.training_model = LLM(
            rank=self.rank,
            world_size=self.world_size,
            model_id=self.model_id,
            model_kwargs=self.model_kwargs,
            parallelize_plan=self.parallelize_plan,
            lora_config=self.lora_config,
            adapter_name=self.adapter_name,
            initialize_random_weights=self.initialize_random_weights,
        )
        self.ready = True
        return True

    async def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> bool:
        config = (
            LoraConfig(**lora_config) if isinstance(lora_config, dict) else lora_config
        )
        self.training_model.add_adapter(adapter_name, config)
        return True

    async def set_active_adapter(self, adapter_name: str) -> bool:
        self.training_model.set_active_adapter(adapter_name)
        return True

    async def get_adapters(self) -> list[str]:
        return list(self.training_model.adapters.keys())

    async def forward(
        self,
        data: list[Datum],
        with_grad: bool = True,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> torch.Tensor:
        try:
            device = torch.cuda.current_device()
            padded = self.training_model.pad(data, device)
            model_inputs = {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
                **padded.get("additional_inputs", {}),
            }
            return self.training_model.forward(
                model_inputs=model_inputs,
                with_grad=with_grad,
                forward_kwargs=forward_kwargs,
            )
        except Exception as e:
            logger.error(f"Forward error: {e}")
            raise

    async def forward_backward(
        self,
        data: list[Datum],
        adapter_name: str | None = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: Dict[str, float] | None = None,
    ):
        if loss_fn not in LOSSES:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        if adapter_name and self.training_model.adapters:
            self.training_model.set_active_adapter(adapter_name)

        device = torch.cuda.current_device()
        padded = self.training_model.pad(data, device)
        logits = self.training_model.forward(
            model_inputs=padded["model_input"],
            with_grad=True,
            forward_kwargs=forward_kwargs,
        )
        per_batch_losses = LOSSES[loss_fn](logits=logits, **padded["loss_fn_inputs"])
        per_batch_losses.mean().backward()

        return (
            {"loss": [loss.item() for loss in per_batch_losses]}
            if self.rank == 0
            else None
        )

    async def save_checkpoint(
        self, checkpoint_path: str, adapter_name: str | None = None
    ):
        if self.rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        self.training_model.save_model(checkpoint_path, adapter_name=adapter_name)
        return self.rank == 0

    async def cleanup(self):
        dist.destroy_process_group()
        return self.rank == 0

    async def zero_grad(self):
        for param in self.training_model.model.parameters():
            param.grad = None

    async def backward(self, loss):
        loss.backward()

    async def optim_step(
        self, adapter_name: str | None = None, optimizer_params: dict[str, Any] = {}
    ):
        if adapter_name and self.training_model.adapters:
            self.training_model.set_active_adapter(adapter_name)
        self._get_optimizer(optimizer_params).step()
