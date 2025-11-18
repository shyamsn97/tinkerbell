import os
import traceback
from typing import Any, Dict

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
from tinkerbell.types.datum import Datum
from tinkerbell.types.loss_fn_type import LossFnType
from tinkerbell.utils import get_submodules_with_wildcard


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
        initialize_random_weights: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.scheduler_params = scheduler_params
        self.initialize_random_weights = initialize_random_weights
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

    def stack_inputs(
        self,
        data: list[Datum],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Stack a list of Datum objects into batched tensors.

        Args:
            data: List of Datum objects to stack

        Returns:
            Tuple of (model_inputs, loss_fn_inputs) where each is a dict of stacked tensors
        """
        try:
            print(f"[stack_inputs] Processing {len(data)} Datum objects")

            # Debug: Check types
            for i, datum in enumerate(data):
                print(f"[stack_inputs] Datum {i}: type={type(datum)}")
                print(
                    f"[stack_inputs] Datum {i}.model_input: type={type(datum.model_input)}"
                )
                print(
                    f"[stack_inputs] Datum {i}.model_input.tokens: type={type(datum.model_input.tokens)}"
                )
                if hasattr(datum.model_input.tokens, "data"):
                    print(
                        f"[stack_inputs] Datum {i}.model_input.tokens.data: {datum.model_input.tokens.data[:5]}..."
                    )
                else:
                    print(
                        f"[stack_inputs] Datum {i}.model_input.tokens: {datum.model_input.tokens}"
                    )

            # Convert all data to torch tensors on CUDA
            device = torch.cuda.current_device()
            print(f"[stack_inputs] Using device: {device}")

            # Stack model inputs
            model_inputs = {}

            # Stack tokens (input_ids for the model)
            print("[stack_inputs] Stacking tokens...")
            tokens_list = [
                datum.model_input.tokens.to_torch(device=device) for datum in data
            ]
            model_inputs["input_ids"] = torch.stack(tokens_list)
            print(
                f"[stack_inputs] Stacked tokens shape: {model_inputs['input_ids'].shape}"
            )

            # Stack attention masks if present
            if data[0].model_input.attention_mask is not None:
                print("[stack_inputs] Stacking attention masks...")
                attention_mask_list = [
                    datum.model_input.attention_mask.to_torch(device=device)
                    for datum in data
                ]
                model_inputs["attention_mask"] = torch.stack(attention_mask_list)
                print(
                    f"[stack_inputs] Stacked attention_mask shape: {model_inputs['attention_mask'].shape}"
                )

            # Stack additional inputs if present
            if data[0].model_input.additional_inputs is not None:
                print("[stack_inputs] Stacking additional inputs...")
                for key in data[0].model_input.additional_inputs.keys():
                    additional_list = [
                        datum.model_input.additional_inputs[key].to_torch(device=device)
                        for datum in data
                    ]
                    model_inputs[key] = torch.stack(additional_list)

            # Stack loss function inputs
            print("[stack_inputs] Stacking loss function inputs...")
            loss_fn_inputs = {}
            if len(data) > 0 and data[0].loss_fn_inputs:
                for key in data[0].loss_fn_inputs.keys():
                    print(f"[stack_inputs] Stacking loss input key: {key}")
                    loss_inputs_list = [
                        datum.loss_fn_inputs[key].to_torch(device=device)
                        for datum in data
                    ]
                    loss_fn_inputs[key] = torch.stack(loss_inputs_list)
                    print(
                        f"[stack_inputs] Stacked {key} shape: {loss_fn_inputs[key].shape}"
                    )

            print("[stack_inputs] Successfully stacked all inputs")
            return model_inputs, loss_fn_inputs
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"[stack_inputs] ERROR: {e}\n{tb_str}")
            raise e

    def setup(self):
        """Initialize the PyTorch distributed process group and model."""
        from transformers import AutoConfig, AutoModelForCausalLM

        if self.ready:
            return True

        self._setup_distributed()

        # Load model
        config = AutoConfig.from_pretrained(self.model_id)
        if self.initialize_random_weights:
            model = AutoModelForCausalLM.from_config(config, **self.model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **self.model_kwargs
            )
        self.model = self._setup_tensor_parallel(model)

        print(f"[Rank {self.rank}] Setup complete")
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
            # Stack inputs from list of Datum objects
            model_inputs, _ = self.stack_inputs(data)
            if with_grad:
                self.model.train()
                outputs = self.model(**model_inputs, **forward_kwargs)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**model_inputs, **forward_kwargs)
            return outputs
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error in forward: {e}\n{tb_str}")
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
        print(f"[Rank {self.rank}] forward_backward called with {len(data)} data items")
        print(f"[Rank {self.rank}] data type: {type(data)}")
        print(
            f"[Rank {self.rank}] first datum type: {type(data[0]) if data else 'empty'}"
        )
        try:
            # Stack inputs from list of Datum objects
            print(f"[Rank {self.rank}] Starting stack_inputs...")
            model_inputs, loss_fn_inputs = self.stack_inputs(data)
            print(f"[Rank {self.rank}] Stacked inputs!")
            # Forward pass
            self.model.train()
            print(f"[Rank {self.rank}] Starting forward pass...")
            outputs = self.model(**model_inputs, **forward_kwargs)
            print(f"[Rank {self.rank}] Forward pass complete!")
            # Compute loss
            if loss_fn == "cross_entropy":
                # Expect 'labels' in loss_fn_inputs
                labels = loss_fn_inputs.get("labels")
                per_batch_losses = ForCausalLMLoss(
                    logits=outputs.logits,
                    labels=labels,
                )
                loss = per_batch_losses.mean()
                print("Loss computed!")
                loss.backward()
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")

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
            self.model.save_pretrained(save_dir, state_dict=state_dict)

        dist.barrier()

    async def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint using PyTorch's distributed checkpoint API.

        Args:
            checkpoint_path: Path to save the checkpoint
        """

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
