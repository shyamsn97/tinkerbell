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

from tinkerbell.training.model import TrainingModel
from tinkerbell.types.datum import Datum
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


class LLM(TrainingModel):

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

    def _pad_sequence(
        self,
        tensors: list[torch.Tensor],
        padding_side: str = "right",
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Pad a list of 1D tensors to the same length.

        Args:
            tensors: List of 1D tensors to pad
            padding_side: 'left' or 'right' padding
            pad_value: Value to use for padding

        Returns:
            Stacked tensor of shape (batch_size, max_length)
        """
        max_len = max(len(t) for t in tensors)
        batch_size = len(tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype

        # Create output tensor filled with pad_value
        padded = torch.full(
            (batch_size, max_len), pad_value, dtype=dtype, device=device
        )

        for i, tensor in enumerate(tensors):
            length = len(tensor)
            if padding_side == "left":
                padded[i, max_len - length :] = tensor
            else:  # right padding
                padded[i, :length] = tensor

        return padded

    def _stack_or_cat_tensors(
        self,
        tensors: list[torch.Tensor],
        padding_side: str = "left",
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Stack or concatenate tensors, handling already-batched and variable-length cases."""
        if len(tensors) == 1:
            # Single tensor - return as-is if already batched (2D+), else add batch dim
            return tensors[0] if tensors[0].ndim >= 2 else tensors[0].unsqueeze(0)

        # Multiple tensors - check if padding needed
        lengths = [t.shape[-1] if t.ndim >= 2 else len(t) for t in tensors]
        if len(set(lengths)) > 1:
            return self._pad_sequence(
                tensors, padding_side=padding_side, pad_value=pad_value
            )
        return torch.cat(tensors) if tensors[0].ndim >= 2 else torch.stack(tensors)

    def stack_inputs(
        self,
        data: list[Datum],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict[str, torch.Tensor]]:
        """Stack Datum objects into batched tensors. Handles both individual and pre-batched inputs."""
        device = torch.cuda.current_device()
        padding_side = data[0].model_input.padding_side or "left"
        pad_token_id = data[0].model_input.pad_token_id or 0

        # Helper to extract and stack field from all data
        def stack_field(field_name: str, pad_value: int = 0) -> torch.Tensor | None:
            tensors = []
            for datum in data:
                val = getattr(datum.model_input, field_name, None)
                if val is None:
                    return None
                tensors.append(val.to_torch(device=device))
            return self._stack_or_cat_tensors(
                tensors=tensors, padding_side=padding_side, pad_value=pad_value
            )

        # Stack model inputs
        model_inputs = {"input_ids": stack_field("tokens", pad_token_id)}

        mask = stack_field("attention_mask", 0)
        if mask is not None:
            model_inputs["attention_mask"] = mask

        labels = stack_field("labels", -100)

        # Stack additional inputs
        if data[0].model_input.additional_inputs:
            for key in data[0].model_input.additional_inputs.keys():
                tensors = [
                    d.model_input.additional_inputs[key].to_torch(device=device)
                    for d in data
                ]
                model_inputs[key] = self._stack_or_cat_tensors(
                    tensors=tensors, padding_side=padding_side, pad_value=0
                )

        # Stack loss function inputs
        loss_fn_inputs = {}
        if data[0].loss_fn_inputs:
            for key in data[0].loss_fn_inputs.keys():
                tensors = [d.loss_fn_inputs[key].to_torch(device=device) for d in data]
                loss_fn_inputs[key] = self._stack_or_cat_tensors(
                    tensors=tensors, padding_side=padding_side, pad_value=0
                )

        return model_inputs, labels, loss_fn_inputs

    def forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        with_grad: bool = True,
        forward_kwargs: dict[str, Any] = {},
    ):
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
            return outputs
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
        from transformers import AutoTokenizer

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
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                tokenizer.save_pretrained(save_dir)
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
