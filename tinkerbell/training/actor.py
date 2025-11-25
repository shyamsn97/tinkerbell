import logging
import os
import traceback
from typing import Any, Dict, Optional

import ray
import torch
import torch.distributed as dist

from tinkerbell.training.llm import LLM
from tinkerbell.training.loss import ForCausalLMLoss
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
        initialize_random_weights: bool = False,
    ):
        # Configure logging for Ray actor - logs will go to stdout
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override any existing configuration
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
        self.training_model = None  # Will be initialized during setup
        self.optimizer = None
        self.ready = False

        # Parse LoRA config
        if lora_config is not None:
            if isinstance(lora_config, dict):
                self.lora_config = LoraConfig(**lora_config)
            else:
                self.lora_config = lora_config
        else:
            self.lora_config = None

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

        # Only optimize parameters that require gradients (important for LoRA)
        trainable_params = [
            p for p in self.training_model.model.parameters() if p.requires_grad
        ]
        optimizer = optimizer_dict[optimizer_name](trainable_params, **optimizer_config)
        return optimizer

    def _setup_distributed(self):
        logger.info(f"[Rank {self.rank}] Initializing torch distributed")

        # Set environment variables for distributed setup
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        # Initialize process group
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

        # Set CUDA device - Ray manages GPU assignment via CUDA_VISIBLE_DEVICES
        # Each actor sees only one GPU as device 0
        torch.cuda.set_device(0)

    def setup(self):
        """Initialize the PyTorch distributed process group and model."""
        if self.ready:
            return True

        self._setup_distributed()

        # Initialize the LLM training model
        # This will create the model, apply LoRA, and parallelize it
        self.training_model = LLM(
            rank=self.rank,
            world_size=self.world_size,
            model_id=self.model_id,
            model_kwargs=self.model_kwargs,
            parallelize_plan=self.parallelize_plan,
            lora_config=self.lora_config,
            initialize_random_weights=self.initialize_random_weights,
        )

        logger.info(f"[Rank {self.rank}] Setup complete")
        self.ready = True
        return True

    async def forward(
        self,
        data: list[Datum],
        with_grad: bool = True,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with automatic tensor conversion from list of Datum objects."""
        try:
            from tinkerbell.training.loss import CROSS_ENTROPY_LOSS_FN
            device = torch.cuda.current_device()
            padded = CROSS_ENTROPY_LOSS_FN.pad(data, device)
            model_inputs = {
                "input_ids": padded["tokens"],
                "attention_mask": padded["attention_mask"],
                **padded.get("additional_inputs", {})
            }
            outputs = self.training_model.forward(
                model_inputs=model_inputs,
                with_grad=with_grad,
                forward_kwargs=forward_kwargs,
            )
            return outputs
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in forward: {e}\n{tb_str}")
            raise e

    async def forward_backward(
        self,
        data: list[Datum],
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: Dict[str, float] | None = None,
    ):
        """Execute a single training step using list of Datum objects."""
        try:
            from tinkerbell.training.loss import CROSS_ENTROPY_LOSS_FN
            
            device = torch.cuda.current_device()
            padded = CROSS_ENTROPY_LOSS_FN.pad(data, device)
            
            model_inputs = {
                "input_ids": padded["tokens"],
                "attention_mask": padded["attention_mask"],
                **padded.get("additional_inputs", {})
            }
            
            outputs = self.training_model.forward(
                model_inputs=model_inputs,
                with_grad=True,
                forward_kwargs=forward_kwargs,
            )

            if loss_fn == "cross_entropy":
                per_batch_losses = ForCausalLMLoss(
                    logits=outputs.logits, **padded["loss_fn_inputs"]
                )
                loss = per_batch_losses.mean()
                loss.backward()
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")

        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in forward_backward: {e}\n{tb_str}")
            raise e

        return (
            {"loss": [per_batch_loss.item() for per_batch_loss in per_batch_losses]}
            if loss is not None and self.rank == 0
            else None
        )

    def save_model(self, save_dir: str):
        """Save the model using the training model's save_model method."""
        self.training_model.save_model(save_dir)

    async def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint using PyTorch's distributed checkpoint API.

        Args:
            checkpoint_path: Path to save the checkpoint
        """

        logger.info(f"[Rank {self.rank}] Starting checkpoint save process...")

        if self.rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)

        self.save_model(checkpoint_path)
        logger.info(f"[Rank {self.rank}] Checkpoint save complete")

        return self.rank == 0

    async def cleanup(self):
        """Clean up the PyTorch distributed process group."""
        logger.info(f"[Rank {self.rank}] Cleaning up torch distributed")
        dist.destroy_process_group()
        return self.rank == 0

    async def zero_grad(self):
        for param in self.training_model.model.parameters():
            param.grad = None

    async def backward(self, loss):
        loss.backward()

    async def optim_step(self, optimizer_params: dict[str, Any] = {}):
        optimizer = self._get_optimizer(optimizer_params)
        optimizer.step()
