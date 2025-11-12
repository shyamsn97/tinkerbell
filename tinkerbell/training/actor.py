import os
import traceback
from typing import Any

import ray
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from tinkerbell.training.loss import ForCausalLMLoss
from tinkerbell.types.data import TensorData
from tinkerbell.utils import get_submodules_with_wildcard


@ray.remote
class TrainingActor:
    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: str,
        model_name: str,
        model_kwargs: dict[str, Any] = {},
        parallelize_plan: dict[str, str] = {},
        scheduler_params: dict[str, Any] = {},
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.scheduler_params = scheduler_params
        self.model = None
        self.optimizer = None
        self.ready = False

    def _get_optimizer(
        self, model: torch.nn.Module, optimizer_config: dict[str, Any]
    ) -> torch.optim.Optimizer:
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
        optimizer = optimizer_dict[optimizer_name](
            model.parameters(), **optimizer_config
        )
        return optimizer

    def _setup_distributed(self):
        print(f"[Rank {self.rank}] Initializing torch distributed")

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

    def _setup_tensor_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        # Define parallelization strategies
        strategies = {
            "column": ColwiseParallel,
            "row": RowwiseParallel,
        }

        # Build module parallelization plan
        module_parallelization_plan = {}
        for pattern in self.parallelize_plan.keys():
            strategy = strategies[self.parallelize_plan[pattern]]()
            module_names = get_submodules_with_wildcard(model, pattern)
            for name in module_names:
                module_parallelization_plan[name] = strategy

        # Initialize device mesh and parallelize model
        device_mesh = init_device_mesh(
            "cuda",
            (
                1,
                self.world_size,
            ),
            mesh_dim_names=(
                "dp",
                "tp",
            ),
        )
        model = parallelize_module(
            model, device_mesh["tp"], module_parallelization_plan
        )
        model = model.cuda()
        return model

    def _prepare_inputs(
        self,
        inputs: list[dict[str, TensorData]],
        targets: dict[str, TensorData] | None = None,
    ) -> dict[str, Any]:
        print("Number of inputs: ", len(inputs))
        print("Inputs: ", inputs)
        print("Targets: ", targets)
        try:
            for input in inputs:
                for key in input:
                    input[key] = input[key].to_torch()
                    input[key] = input[key].cuda()
            if targets is not None:
                targets = torch.stack([t.to_torch() for t in targets])
                targets = targets.cuda()
            print("Torch inputs: ", inputs)
            print("Torch targets: ", targets)
            batch_inputs = {}
            for key in inputs[0]:
                batch_inputs[key] = torch.stack(
                    [input[key] for input in inputs]
                ).squeeze(0)
            print(f"Actor Batch inputs: {batch_inputs}")
            print("Input shapes:")
            for key in batch_inputs:
                print(f"  - {key}: {batch_inputs[key].shape}")
            print("Torch targets shape: ", targets.shape)
            return batch_inputs, targets
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in _prepare_inputs: {e}\n{tb_str}")
            raise e

    def setup(self):
        """Initialize the PyTorch distributed process group and model."""
        from transformers import AutoConfig, AutoModelForCausalLM

        if self.ready:
            return True

        self._setup_distributed()

        # Load model
        config = AutoConfig.from_pretrained(self.model_name, **self.model_kwargs)
        model = AutoModelForCausalLM.from_config(config)
        self.model = self._setup_tensor_parallel(model)

        print(f"[Rank {self.rank}] Setup complete")
        self.ready = True
        return True

    async def forward(
        self,
        batch_inputs: dict[str, torch.Tensor],
        with_grad: bool = True,
        forward_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with automatic tensor conversion from lists/arrays."""
        # Convert inputs to tensors if they're not already
        try:
            if with_grad:
                self.model.train()
                outputs = self.model(**batch_inputs, **forward_kwargs)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**batch_inputs, **forward_kwargs)
            print(f"Actor Outputs logits shape {self.rank}: {outputs.logits.shape}")
            return outputs
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in forward: {e}\n{tb_str}")
            raise e

    async def forward_backward(
        self,
        inputs: list[dict[str, TensorData]],
        targets: Any,
        forward_kwargs: dict[str, Any] = {},
        return_logprobs: bool = False,
        **kwargs,
    ):
        """Execute a single training step (accepts tensors or lists from JSON)."""
        batch_inputs, targets = self._prepare_inputs(inputs, targets)
        outputs = await self.forward(
            batch_inputs=batch_inputs,
            with_grad=True,
            forward_kwargs=forward_kwargs,
        )
        try:
            per_batch_losses = ForCausalLMLoss(
                logits=outputs.logits,
                labels=targets,
            )
            loss = per_batch_losses.mean()
            loss.backward()
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in forward_backward: {e}\n{tb_str}")
            raise e

        return (
            {"loss": [per_batch_loss.item() for per_batch_loss in per_batch_losses]}
            if loss is not None and self.rank == 0
            else None
        )

    def get_model_state_dict(self, full_state_dict: bool = False):
        options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=True)
        # self._load_model_to_device(torch.cuda.current_device())
        state_dict = get_model_state_dict(self.model, options=options)
        # self._load_model_to_device("cpu")
        return state_dict

    def save_model(self, save_dir: str):

        state_dict = self.get_model_state_dict(full_state_dict=True)
        if self.rank == 0:

            # self.tokenizer.save_pretrained(save_dir)
            self.model.save_pretrained(save_dir, state_dict=state_dict)

        dist.barrier()

    async def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint using PyTorch's distributed checkpoint API.

        Args:
            checkpoint_path: Path to save the checkpoint
        """
        # from torch.distributed.checkpoint import save

        print(f"[Rank {self.rank}] Starting checkpoint save process...")

        if self.rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)

        self.save_model(checkpoint_path)
        print(f"[Rank {self.rank}] Checkpoint save complete")

        return self.rank == 0

    async def cleanup(self):
        """Clean up the PyTorch distributed process group."""
        print(f"[Rank {self.rank}] Cleaning up torch distributed")
        dist.destroy_process_group()
        return self.rank == 0

    async def zero_grad(self):
        for param in self.model.parameters():
            param.grad = None

    async def backward(self, loss):
        loss.backward()

    async def optim_step(self, optimizer_params: dict[str, Any] = {}):
        optimizer = self._get_optimizer(self.model, optimizer_params)
        optimizer.step()
