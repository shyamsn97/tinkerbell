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
        base_model: str,
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
        self.base_model = base_model
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
        # Make a copy to avoid mutating the original dict
        optimizer_config = optimizer_config.copy()
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

    def _get_or_create_optimizer(
        self, optimizer_config: dict[str, Any]
    ) -> torch.optim.Optimizer:
        """Get existing optimizer or create a new one if trainable parameters changed."""
        # Get current trainable parameters
        current_trainable_params = [
            p for p in self.training_model.model.parameters() if p.requires_grad
        ]
        current_param_ids = {id(p) for p in current_trainable_params}

        # Check if we need to create a new optimizer
        if self.optimizer is None:
            self.optimizer = self._get_optimizer(optimizer_config)
            self._last_param_ids = current_param_ids
        else:
            # Check if trainable parameters changed (e.g., adapter switch)
            if (
                not hasattr(self, "_last_param_ids")
                or self._last_param_ids != current_param_ids
            ):
                # Parameters changed, need to recreate optimizer
                logger.info("Trainable parameters changed, recreating optimizer")
                del self.optimizer
                self.optimizer = self._get_optimizer(optimizer_config)
                self._last_param_ids = current_param_ids
            else:
                # Update learning rate if it changed
                if "lr" in optimizer_config:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = optimizer_config["lr"]

        return self.optimizer

    def _setup_distributed(self):
        # Set PyTorch CUDA allocator to reduce memory fragmentation
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
            base_model=self.base_model,
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
        forward_kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
        try:
            device = torch.cuda.current_device()
            padded = self.training_model.pad(data, device)
            forward_output = self.training_model.forward(
                model_inputs=padded["model_input"],
                with_grad=True,
                forward_kwargs=forward_kwargs,
            )
            return forward_output["logprobs"]
        except Exception as e:
            logger.error(f"Forward error: {e}")
            raise

    async def forward_backward(
        self,
        data: list[Datum],
        adapter_name: str | None = None,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
        loss_fns: list[LossFnType] = ["cross_entropy"],
        loss_fn_config: Dict[str, float] | None = None,
    ):
        try:
            for loss_fn in loss_fns:
                if loss_fn not in LOSSES:
                    raise ValueError(f"Unknown loss function: {loss_fn}")

            if len(loss_fns) != len(data):
                raise ValueError(
                    f"Number of loss functions must match number of data points: {len(loss_fns)} != {len(data)}"
                )

            if adapter_name and self.training_model.adapters:
                self.training_model.set_active_adapter(adapter_name)

            # Safety check: warn if gradients exist (but don't clear to support gradient accumulation)
            # Only iterate over trainable parameters (LoRA adapters), not base model
            has_existing_grads = False
            for param in (
                p for p in self.training_model.model.parameters() if p.requires_grad
            ):
                if param.grad is not None:
                    has_existing_grads = True
                    break
            if has_existing_grads:
                logger.debug(
                    "Found existing gradients before forward_backward - this is expected for gradient accumulation"
                )

            self.training_model.train()
            device = torch.cuda.current_device()
            padded = self.training_model.pad(data, device)
            forward_output = self.training_model.forward(
                model_inputs=padded["model_input"],
                with_grad=True,
                forward_kwargs=forward_kwargs,
            )
            logprobs = forward_output["logprobs"]

            # Check if all loss functions are the same
            if len(set(loss_fns)) == 1:
                loss_fn = loss_fns[0]
                per_batch_losses = LOSSES[loss_fn](
                    logprobs=logprobs, **padded["loss_fn_inputs"]
                )
            else:
                per_batch_losses = torch.stack(
                    [
                        LOSSES[loss_fn](
                            logprobs=logprobs[i : i + 1],
                            **{
                                k: v[i : i + 1]
                                for k, v in padded["loss_fn_inputs"].items()
                            },
                        )
                        for i, loss_fn in enumerate(loss_fns)
                    ]
                ).squeeze()

            loss_mean = per_batch_losses.mean()
            loss_mean.backward()

            # Extract loss values before cleaning up tensors
            loss_values = (
                [loss.item() for loss in per_batch_losses] if self.rank == 0 else None
            )

            # Clean up intermediate tensors to free memory
            del forward_output, padded, per_batch_losses, loss_mean

            sum_gradient = {}
            with torch.no_grad():
                for name, param in self.training_model.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Use detach() to ensure we're not creating references
                        # Only compute for parameters that actually have gradients
                        grad_sum = param.grad.detach().sum().item()
                        sum_gradient[name] = grad_sum

            if self.rank == 0:
                return {
                    "loss": loss_values,
                    "sum_gradient": sum_gradient,
                }
            else:
                return None
        except torch.cuda.OutOfMemoryError as e:
            # Clear cache and re-raise with more context
            torch.cuda.empty_cache()
            logger.error(
                f"CUDA out of memory error in forward_backward (rank {self.rank}): {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"CUDA out of memory during forward_backward: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(
                f"Error in forward_backward (rank {self.rank}): {e}", exc_info=True
            )
            raise

    async def save_checkpoint(
        self, checkpoint_path: str, adapter_name: str | None = None
    ):
        if self.rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        self.training_model.save_model(checkpoint_path, adapter_name=adapter_name)
        return self.rank == 0

    async def push_to_hub(
        self,
        repo_id: str,
        adapter_name: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
        push_kwargs: dict[str, Any] = {},
    ):
        """Push model to Hugging Face Hub.

        The repository will be created automatically if it doesn't exist.
        Requires authentication via token or huggingface_hub login.
        """
        if self.rank == 0:
            if adapter_name and self.training_model.adapters:
                self.training_model.set_active_adapter(adapter_name)

            if not hasattr(self.training_model.model, "push_to_hub"):
                raise RuntimeError("Model does not support push_to_hub")

            push_params = {
                "repo_id": repo_id,
                "token": token,
                "private": private,
                "commit_message": commit_message,
                **push_kwargs,
            }
            if adapter_name and self.training_model.adapters:
                push_params["adapter_name"] = adapter_name

            self.training_model.model.push_to_hub(**push_params)

            # Push tokenizer if available
            if self.training_model.tokenizer is not None:
                try:
                    self.training_model.tokenizer.push_to_hub(
                        repo_id=repo_id,
                        token=token,
                        private=private,
                        commit_message=commit_message,
                    )
                except Exception as e:
                    logger.warning(f"Failed to push tokenizer: {e}")

        # Wait for all ranks to complete
        dist.barrier()
        return self.rank == 0

    async def cleanup(self):
        dist.destroy_process_group()
        return self.rank == 0

    async def zero_grad(self):
        """Clear all gradients."""
        if self.training_model is None or self.training_model.model is None:
            logger.warning("Model not initialized yet, skipping zero_grad")
            return

        # Only clear gradients for trainable parameters (LoRA adapters), not base model
        for param in (
            p for p in self.training_model.model.parameters() if p.requires_grad
        ):
            if param.grad is not None:
                param.grad = None

    async def backward(self, loss):
        loss.backward()

    async def optim_step(
        self, adapter_name: str | None = None, optimizer_params: dict[str, Any] = {}
    ):
        if adapter_name and self.training_model.adapters:
            self.training_model.set_active_adapter(adapter_name)

        # Reuse existing optimizer or create/update if needed
        optimizer = self._get_or_create_optimizer(optimizer_params)
        optimizer.step()

        # Clear gradients after optimizer step
        # Only clear gradients for trainable parameters (LoRA adapters), not base model
        for param in (
            p for p in self.training_model.model.parameters() if p.requires_grad
        ):
            if param.grad is not None:
                param.grad = None
