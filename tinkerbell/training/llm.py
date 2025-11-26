import logging
import re
import traceback
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
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

from tinkerbell.types.lora_config import LoraConfig
from tinkerbell.utils import get_submodules_with_wildcard

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

logger = logging.getLogger(__name__)


class LLM:

    def __init__(
        self,
        rank: int,
        world_size: int,
        model_id: str,
        model_kwargs: dict[str, Any],
        parallelize_plan: dict[str, str],
        lora_config: LoraConfig | None = None,
        initialize_random_weights: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.lora_config = lora_config
        self.initialize_random_weights = initialize_random_weights
        self.should_merge_lora = False
        self.tokenizer = None  # Will be initialized during setup
        self.setup()

    def setup(self) -> None:
        """
        Setup the model, tokenizer, and padding strategies.
        """
        from transformers import AutoTokenizer

        from tinkerbell.types.data import PaddingStrategy

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Setup padding strategies using tokenizer's pad_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or 0

        pad_token_id = self.tokenizer.pad_token_id

        self.padding_strategies = {
            "model_input.input_ids": PaddingStrategy(
                padding_side="left", padding_value=pad_token_id
            ),
            "model_input.attention_mask": PaddingStrategy(
                padding_side="left", padding_value=0
            ),
            "loss_fn_inputs.labels": PaddingStrategy(
                padding_side="left", padding_value=-100
            ),
        }

        # Load and configure model
        self.model = self.create_model(
            model_id=self.model_id,
            model_kwargs=self.model_kwargs,
        )
        self.model = self.setup_lora(
            model=self.model,
            lora_config=self.lora_config,
        )
        self.model = self.parallelize(
            model=self.model,
            parallelize_plan=self.parallelize_plan,
        )

    def create_model(self, model_id: str, model_kwargs: dict[str, Any]) -> nn.Module:
        from transformers import AutoConfig, AutoModelForCausalLM

        # Load model
        config = AutoConfig.from_pretrained(model_id)
        if self.initialize_random_weights:
            model = AutoModelForCausalLM.from_config(config, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **self.model_kwargs
            )
        return model

    def setup_lora(
        self, model: nn.Module, lora_config: LoraConfig | None = None
    ) -> nn.Module:
        if lora_config is None:
            return model
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA support. "
                "Install it with: pip install peft"
            )

        logger.info(f"[Rank {self.rank}] Applying LoRA with config: {lora_config}")

        # Build target modules list based on config
        target_modules = []

        if lora_config.train_attn:
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

        if lora_config.train_mlp:
            # Standard MLP module names
            target_modules.extend(
                [
                    "gate_proj",
                    "up_proj",
                    "down_proj",  # LLaMA, Mistral
                    "gate_up_proj",  # Some architectures combine gate and up
                ]
            )

        if lora_config.train_unembed:
            target_modules.extend(
                [
                    "lm_head",  # Standard language model head
                    "embed_out",  # Alternative naming
                ]
            )

        # Create PEFT LoRA config
        peft_config = PeftLoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.rank * 2,  # Common default: 2x rank
            target_modules=target_modules,
            lora_dropout=0.0,  # Can be made configurable if needed
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)

        def matches_supported_pattern(target_module: str) -> bool:
            """Check if target_module contains any supported pattern."""
            for supported in SUPPORTED_LORA_TARGET_MODULES:
                # Use regex to check if supported pattern appears in target_module
                # This handles cases like "attention1.*.gate_proj.0" matching "gate_proj"
                pattern = re.escape(supported)
                if re.search(pattern, target_module):
                    return True
            return False

        all_supported = all(
            matches_supported_pattern(target_module) for target_module in target_modules
        )

        if not all_supported:
            unsupported = [
                tm for tm in target_modules if not matches_supported_pattern(tm)
            ]
            logger.warning(
                f"[Rank {self.rank}] LoRA target modules {unsupported} do not match "
                f"supported patterns {SUPPORTED_LORA_TARGET_MODULES}. "
                f"Will merge and save as full model."
            )
            self.should_merge_lora = True

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

    def pad(
        self,
        data: list,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Pad a batch of Datum objects and return dictionary of padded tensors.

        Args:
            data: List of Datum objects
            device: Device to place tensors on

        Returns:
            Dictionary with padded tensors using nested paths from padding strategies
        """
        from tinkerbell.utils import get_nested, set_nested

        torch_data = [d.to_torch(device=device) for d in data]
        result = {}

        # Iterate through all padding strategy keys and apply them
        for path, padding_strategy in self.padding_strategies.items():
            # Extract values from each datum using the nested path
            values = [get_nested(d, path) for d in torch_data]

            # Handle None values - skip padding if all values are None
            if all(v is None for v in values):
                continue

            # If some values are None, we need to handle them
            # For attention_mask, None means all tokens are valid, so create all-ones mask
            if any(v is None for v in values):
                if "attention_mask" in path:
                    # Get the token lengths to create proper attention masks
                    input_ids_path = path.replace("attention_mask", "input_ids")
                    input_ids_values = [
                        get_nested(d, input_ids_path) for d in torch_data
                    ]
                    values = [
                        torch.ones_like(input_ids_values[i]) if v is None else v
                        for i, v in enumerate(values)
                    ]
                else:
                    # For other fields, skip None values (shouldn't happen for input_ids/labels)
                    continue

            # Pad the values
            padded = padding_strategy.pad_sequence(values)

            # Set the padded result back using the nested path
            set_nested(result, path, padded)

        return result

    def forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        with_grad: bool = True,
        forward_kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            model_inputs: Dictionary of model inputs
            with_grad: Whether to enable gradient computation
            forward_kwargs: Additional kwargs to pass to the model

        Returns:
            Model outputs
        """
        try:
            if with_grad:
                self.model.train()
                outputs = self.model(**model_inputs, **forward_kwargs)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**model_inputs, **forward_kwargs)
            return outputs.logits
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"Error in forward: {e}\n{tb_str}")
            raise e

    def get_model_state_dict(self, full_state_dict: bool = False):
        """Get the model state dict.

        Args:
            full_state_dict: Whether to return the full state dict (for distributed models)

        Returns:
            Model state dict
        """
        options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)
        return state_dict

    def save_model(self, save_dir: str):
        """Save the model to a directory.

        Args:
            save_dir: Directory to save the model to
        """
        if self.rank == 0:

            # Check if this is a PEFT model
            if self.lora_config is not None:
                # Check if we need to merge LoRA weights due to unsupported target modules
                if self.should_merge_lora:
                    try:
                        # Merge LoRA weights into the base model
                        self.model = self.model.merge_and_unload()
                        # Save as full model
                        state_dict = self.get_model_state_dict(full_state_dict=True)
                        self.model.save_pretrained(save_dir, state_dict=state_dict)
                    except Exception as e:
                        logger.error(f"Failed to merge LoRA weights: {e}")
                        raise
                else:
                    # For LoRA models with supported target modules, save ONLY the adapter weights (lightweight!)
                    self.model.save_pretrained(save_dir)
            else:
                # For full fine-tuning, save the full model
                state_dict = self.get_model_state_dict(full_state_dict=True)
                self.model.save_pretrained(save_dir, state_dict=state_dict)

            # Also save the tokenizer - SGLang needs it to load the model
            try:
                self.tokenizer.save_pretrained(save_dir)
            except Exception:
                logger.warning("Warning: Failed to save tokenizer")
                logger.warning("The checkpoint may not be loadable by SGLang")

        dist.barrier()

    def parallelize(
        self, model: nn.Module, parallelize_plan: dict[str, str]
    ) -> nn.Module:
        """Apply tensor parallelism to the model.

        Args:
            model: Model to parallelize
            parallelize_plan: Dictionary mapping module patterns to parallelization strategies

        Returns:
            Parallelized model
        """
        if not parallelize_plan:
            # No parallelization requested, just move to GPU
            return model.cuda()

        # Define parallelization strategies
        strategies = {
            "column": ColwiseParallel,
            "row": RowwiseParallel,
        }

        # Build module parallelization plan
        module_parallelization_plan = {}
        for pattern in parallelize_plan.keys():
            strategy = strategies[parallelize_plan[pattern]]()
            module_names = get_submodules_with_wildcard(model, pattern)
            for name in module_names:
                module_parallelization_plan[name] = strategy

        # Initialize device mesh and parallelize model
        device_mesh = init_device_mesh(
            "cuda",
            (1, self.world_size),
            mesh_dim_names=("dp", "tp"),
        )
        model = parallelize_module(
            model, device_mesh["tp"], module_parallelization_plan
        )
        model = model.cuda()
        return model
