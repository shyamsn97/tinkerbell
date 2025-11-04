import os
import traceback
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from tinkerbell.utils import get_submodules_with_wildcard


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
        optimizer_params: dict[str, Any] = {},
        scheduler_params: dict[str, Any] = {},
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.model = None
        self.optimizer = None

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
        optimizer = optimizer_dict[optimizer_name](
            self.model.parameters(), **optimizer_config
        )
        return optimizer

    def setup(self):
        """Initialize the PyTorch distributed process group and model."""
        from transformers import AutoConfig, AutoModelForCausalLM

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

        # Load model
        config = AutoConfig.from_pretrained(self.model_name, **self.model_kwargs)
        self.model = AutoModelForCausalLM.from_config(config)

        # Define parallelization strategies
        strategies = {
            "column": ColwiseParallel,
            "row": RowwiseParallel,
        }

        # Build module parallelization plan
        module_parallelization_plan = {}
        for pattern in self.parallelize_plan.keys():
            strategy = strategies[self.parallelize_plan[pattern]]()
            module_names = get_submodules_with_wildcard(self.model, pattern)
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
        self.model = parallelize_module(
            self.model, device_mesh["tp"], module_parallelization_plan
        )
        self.model = self.model.cuda()

        # # Print memory - clarify that each actor uses device 0 (Ray's CUDA_VISIBLE_DEVICES isolation)
        # print_gpu_memory(f"Model loaded (Rank {self.rank}, physical device isolated by Ray as cuda:0)", 0)

        # Setup optimizer - disable foreach to handle mixed DTensor/Tensor parameters
        self.optimizer = self._get_optimizer(self.optimizer_params)

        print(f"[Rank {self.rank}] Setup complete")
        return True

    def zero_grad(self):
        self.optimizer.zero_grad()

    def backward(self, loss):
        loss.backward()

    def step(self):
        self.optimizer.step()

    async def forward(
        self, inputs: list[dict[str, Any]], targets: Any = None, **kwargs
    ):
        """Forward pass with automatic tensor conversion from lists/arrays."""
        # Convert inputs to tensors if they're not already
        try:
            print("Number of inputs: ", len(inputs))
            for input in inputs:
                for key in input:
                    if isinstance(input[key], (list, tuple)):
                        # Convert from JSON-deserialized lists back to tensors
                        input[key] = torch.tensor(input[key])
                    if isinstance(input[key], torch.Tensor):
                        input[key] = input[key].cuda()

            if targets is not None:
                if isinstance(targets, (list, tuple)):
                    targets = torch.tensor(targets)
                if isinstance(targets, torch.Tensor):
                    targets = targets.cuda()

            batch_inputs = {}
            for key in inputs[0]:
                if isinstance(inputs[0][key], torch.Tensor):
                    batch_inputs[key] = torch.stack(
                        [input[key] for input in inputs]
                    ).squeeze(0)
                else:
                    batch_inputs[key] = [input[key] for input in inputs]

            if targets is not None:
                batch_inputs["labels"] = targets

            print(f"Actor Batch inputs: {batch_inputs}")
            print(f"Actor Forward kwargs: {kwargs}")
            print("Input shapes:")
            for key in batch_inputs:
                print(f"  - {key}: {batch_inputs[key].shape}")

            self.model.train()
            outputs = self.model(**batch_inputs, **kwargs)
            print(f"Actor Outputs: {outputs}")
            return outputs
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in forward: {e}\n{tb_str}")
            raise e

    async def forward_backward(
        self, inputs: list[dict[str, Any]], targets: Any = None, **kwargs
    ):
        """Execute a single training step (accepts tensors or lists from JSON)."""
        # Prepare data
        # tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # tokenizer.pad_token = tokenizer.eos_token
        # inputs = tokenizer(["Hello world!"], return_tensors="pt", padding=True)
        # input_ids = inputs["input_ids"].cuda()

        # Training step
        outputs = await self.forward(inputs, targets, **kwargs)
        loss = outputs.loss

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        return {"loss": loss.item() if loss is not None and self.rank == 0 else None}

    def cleanup(self):
        """Clean up the PyTorch distributed process group."""
        print(f"[Rank {self.rank}] Cleaning up torch distributed")
        dist.destroy_process_group()
        return True
