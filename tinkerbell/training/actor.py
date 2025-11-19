import logging
import os
import traceback
from typing import Any, Dict, Optional

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
from tinkerbell.types.lora_config import LoraConfig
from tinkerbell.types.loss_fn_type import LossFnType
from tinkerbell.utils import get_submodules_with_wildcard

logger = logging.getLogger(__name__)

SUPPORTED_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "qkv_proj",
    "gate_up_proj",
]


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

        # Parse LoRA config
        if lora_config is not None:
            if isinstance(lora_config, dict):
                self.lora_config = LoraConfig(**lora_config)
            else:
                self.lora_config = lora_config
        else:
            self.lora_config = None

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

        # Only optimize parameters that require gradients (important for LoRA)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
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

    def _setup_lora(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply LoRA to the model using PEFT."""
        if self.lora_config is None:
            return model

        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA support. "
                "Install it with: pip install peft"
            )

        logger.info(f"[Rank {self.rank}] Applying LoRA with config: {self.lora_config}")

        # Build target modules list based on config
        target_modules = []

        if self.lora_config.train_attn:
            # Standard attention module names across different architectures
            target_modules.extend(
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",  # LLaMA, Mistral, etc.
                    "qkv_proj",  # Some architectures use combined QKV
                ]
            )

        if self.lora_config.train_mlp:
            # Standard MLP module names
            target_modules.extend(
                [
                    "gate_proj",
                    "up_proj",
                    "down_proj",  # LLaMA, Mistral
                    "gate_up_proj",  # Some architectures combine gate and up
                ]
            )

        if self.lora_config.train_unembed:
            target_modules.extend(
                [
                    "lm_head",  # Standard language model head
                    "embed_out",  # Alternative naming
                ]
            )

        # Create PEFT LoRA config
        peft_config = PeftLoraConfig(
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.rank * 2,  # Common default: 2x rank
            target_modules=target_modules,
            lora_dropout=0.0,  # Can be made configurable if needed
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)

        # Print trainable parameters info
        if self.rank == 0:
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"[Rank {self.rank}] LoRA applied successfully!")
            logger.info(
                f"[Rank {self.rank}] Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

        return model

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
            logger.debug(f"[stack_inputs] Processing {len(data)} Datum objects")

            # Debug: Check types
            for i, datum in enumerate(data):
                logger.debug(f"[stack_inputs] Datum {i}: type={type(datum)}")
                logger.debug(
                    f"[stack_inputs] Datum {i}.model_input: type={type(datum.model_input)}"
                )
                logger.debug(
                    f"[stack_inputs] Datum {i}.model_input.tokens: type={type(datum.model_input.tokens)}"
                )
                if hasattr(datum.model_input.tokens, "data"):
                    logger.debug(
                        f"[stack_inputs] Datum {i}.model_input.tokens.data: {datum.model_input.tokens.data[:5]}..."
                    )
                else:
                    logger.debug(
                        f"[stack_inputs] Datum {i}.model_input.tokens: {datum.model_input.tokens}"
                    )

            # Convert all data to torch tensors on CUDA
            device = torch.cuda.current_device()
            logger.debug(f"[stack_inputs] Using device: {device}")

            # Stack model inputs
            model_inputs = {}

            # Stack tokens (input_ids for the model)
            logger.debug("[stack_inputs] Stacking tokens...")
            tokens_list = [
                datum.model_input.tokens.to_torch(device=device) for datum in data
            ]
            model_inputs["input_ids"] = torch.stack(tokens_list)
            logger.debug(
                f"[stack_inputs] Stacked tokens shape: {model_inputs['input_ids'].shape}"
            )

            # Stack attention masks if present
            if data[0].model_input.attention_mask is not None:
                logger.debug("[stack_inputs] Stacking attention masks...")
                attention_mask_list = [
                    datum.model_input.attention_mask.to_torch(device=device)
                    for datum in data
                ]
                model_inputs["attention_mask"] = torch.stack(attention_mask_list)
                logger.debug(
                    f"[stack_inputs] Stacked attention_mask shape: {model_inputs['attention_mask'].shape}"
                )

            # Stack additional inputs if present
            if data[0].model_input.additional_inputs is not None:
                logger.debug("[stack_inputs] Stacking additional inputs...")
                for key in data[0].model_input.additional_inputs.keys():
                    additional_list = [
                        datum.model_input.additional_inputs[key].to_torch(device=device)
                        for datum in data
                    ]
                    model_inputs[key] = torch.stack(additional_list)

            # Stack loss function inputs
            logger.debug("[stack_inputs] Stacking loss function inputs...")
            loss_fn_inputs = {}
            if len(data) > 0 and data[0].loss_fn_inputs:
                for key in data[0].loss_fn_inputs.keys():
                    logger.debug(f"[stack_inputs] Stacking loss input key: {key}")
                    loss_inputs_list = [
                        datum.loss_fn_inputs[key].to_torch(device=device)
                        for datum in data
                    ]
                    loss_fn_inputs[key] = torch.stack(loss_inputs_list)
                    logger.debug(
                        f"[stack_inputs] Stacked {key} shape: {loss_fn_inputs[key].shape}"
                    )

            logger.debug("[stack_inputs] Successfully stacked all inputs")
            return model_inputs, loss_fn_inputs
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"[stack_inputs] ERROR: {e}\n{tb_str}")
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

        # Apply LoRA if configured (before tensor parallelism)
        model = self._setup_lora(model)

        # Apply tensor parallelism
        self.model = self._setup_tensor_parallel(model)

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
        logger.debug(
            f"[Rank {self.rank}] forward_backward called with {len(data)} data items"
        )
        logger.debug(f"[Rank {self.rank}] data type: {type(data)}")
        logger.debug(
            f"[Rank {self.rank}] first datum type: {type(data[0]) if data else 'empty'}"
        )
        try:
            # Stack inputs from list of Datum objects
            logger.debug(f"[Rank {self.rank}] Starting stack_inputs...")
            model_inputs, loss_fn_inputs = self.stack_inputs(data)
            logger.debug(f"[Rank {self.rank}] Stacked inputs!")
            # Forward pass
            self.model.train()
            logger.debug(f"[Rank {self.rank}] Starting forward pass...")
            outputs = self.model(**model_inputs, **forward_kwargs)
            logger.debug(f"[Rank {self.rank}] Forward pass complete!")
            # Compute loss
            if loss_fn == "cross_entropy":
                # Expect 'labels' in loss_fn_inputs
                labels = loss_fn_inputs.get("labels")
                per_batch_losses = ForCausalLMLoss(
                    logits=outputs.logits,
                    labels=labels,
                )
                loss = per_batch_losses.mean()
                logger.debug("Loss computed!")
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

    def get_model_state_dict(self, full_state_dict: bool = False):
        options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=True)
        # self._load_model_to_device(torch.cuda.current_device())
        state_dict = get_model_state_dict(self.model, options=options)
        # self._load_model_to_device("cpu")
        return state_dict

    def save_model(self, save_dir: str):
        from transformers import AutoTokenizer

        if self.rank == 0:
            logger.info(f"[Rank {self.rank}] Saving model to {save_dir}")

            # Check if this is a PEFT model
            if self.lora_config is not None:
                # For LoRA models, save ONLY the adapter weights (lightweight!)
                logger.info(f"[Rank {self.rank}] Saving LoRA adapter weights")
                self.model.save_pretrained(save_dir)

                # Save config to indicate this is a LoRA adapter
                # import json

                # lora_info = {
                #     "is_lora_adapter": True,
                #     "base_model_id": self.model_id,
                #     "lora_config": self.lora_config.model_dump(),
                # }
                # with open(
                #     os.path.join(save_dir, "tinkerbell_lora_info.json"), "w"
                # ) as f:
                #     json.dump(lora_info, f, indent=2)
            else:
                # For full fine-tuning, save the full model
                state_dict = self.get_model_state_dict(full_state_dict=True)
                self.model.save_pretrained(save_dir, state_dict=state_dict)

            # Also save the tokenizer - SGLang needs it to load the model
            logger.info(f"[Rank {self.rank}] Saving tokenizer to {save_dir}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                tokenizer.save_pretrained(save_dir)
                logger.info(f"[Rank {self.rank}] Tokenizer saved successfully")
            except Exception as e:
                logger.warning(
                    f"[Rank {self.rank}] Warning: Failed to save tokenizer: {e}"
                )
                logger.warning(
                    f"[Rank {self.rank}] The checkpoint may not be loadable by SGLang"
                )

        dist.barrier()

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
        for param in self.model.parameters():
            param.grad = None

    async def backward(self, loss):
        loss.backward()

    async def optim_step(self, optimizer_params: dict[str, Any] = {}):
        optimizer = self._get_optimizer(self.model, optimizer_params)
        optimizer.step()
