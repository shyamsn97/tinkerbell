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
        adapter_name: str | None = None,
        initialize_random_weights: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.lora_config = lora_config
        self.adapter_name = adapter_name or "default"
        self.initialize_random_weights = initialize_random_weights
        self.should_merge_lora = {}  # Per-adapter merge flags
        self.adapters: dict[str, LoraConfig] = {}  # Track all adapters
        self.active_adapter: str | None = None
        self.tokenizer = None
        self.setup()

    def setup(self) -> None:
        """Setup the model, tokenizer, and padding strategies."""
        from transformers import AutoTokenizer

        from tinkerbell.types.data import PaddingStrategy

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
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

        self.model = self.create_model(
            model_id=self.model_id, model_kwargs=self.model_kwargs
        )
        if self.lora_config is not None:
            self.model = self.add_adapter(self.adapter_name, self.lora_config)
        self.model = self.parallelize(
            model=self.model, parallelize_plan=self.parallelize_plan
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

    def _build_target_modules(self, lora_config: LoraConfig) -> list[str]:
        """Build target modules list from LoRA config."""
        target_modules = []
        if lora_config.train_attn:
            target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"])
        if lora_config.train_mlp:
            target_modules.extend(["gate_proj", "up_proj", "down_proj", "gate_up_proj"])
        if lora_config.train_unembed:
            target_modules.extend(["lm_head", "embed_out"])
        return target_modules

    def add_adapter(self, adapter_name: str, lora_config: LoraConfig) -> nn.Module:
        """Add a named LoRA adapter. Supports multiple adapters on same base model."""
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        if adapter_name in self.adapters:
            logger.info(
                f"[Rank {self.rank}] Adapter '{adapter_name}' already exists, switching to it"
            )
            self.set_active_adapter(adapter_name)
            return self.model

        logger.info(
            f"[Rank {self.rank}] Adding adapter '{adapter_name}' with config: {lora_config}"
        )
        target_modules = self._build_target_modules(lora_config)

        peft_config = PeftLoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.rank * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # First adapter: wrap with get_peft_model; subsequent: add_adapter
        if not self.adapters:
            self.model = get_peft_model(
                self.model, peft_config, adapter_name=adapter_name
            )
        else:
            self.model.add_adapter(adapter_name, peft_config)

        self.adapters[adapter_name] = lora_config
        self.set_active_adapter(adapter_name)

        # Check for unsupported target modules
        unsupported = [
            tm
            for tm in target_modules
            if not any(
                re.search(re.escape(s), tm) for s in SUPPORTED_LORA_TARGET_MODULES
            )
        ]
        self.should_merge_lora[adapter_name] = bool(unsupported)
        if unsupported:
            logger.warning(
                f"[Rank {self.rank}] Adapter '{adapter_name}' has unsupported modules {unsupported}"
            )

        if self.rank == 0:
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"[Rank {self.rank}] Adapter '{adapter_name}' added. Trainable: {trainable:,}/{total:,}"
            )

        return self.model

    def set_active_adapter(self, adapter_name: str) -> None:
        """Set which adapter is active for training/inference."""
        if adapter_name not in self.adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {list(self.adapters.keys())}"
            )
        self.model.set_adapter(adapter_name)
        self.active_adapter = adapter_name
        logger.info(f"[Rank {self.rank}] Active adapter set to '{adapter_name}'")

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

    def save_model(self, save_dir: str, adapter_name: str | None = None):
        """Save model or specific adapter to a directory."""
        import os

        # For full model (no adapters): get_model_state_dict needs ALL ranks
        state_dict = None
        if not self.adapters:
            state_dict = self.get_model_state_dict(full_state_dict=True)

        if self.rank == 0:
            import shutil

            os.makedirs(save_dir, exist_ok=True)
            if self.adapters:
                name = adapter_name or self.active_adapter
                if name and self.should_merge_lora.get(name, False):
                    self.model = self.model.merge_and_unload()
                    merged_state = self.get_model_state_dict(full_state_dict=True)
                    self.model.save_pretrained(save_dir, state_dict=merged_state)
                else:
                    # Save adapter - PEFT creates files in save_dir/adapter_name/
                    self.model.save_pretrained(
                        save_dir, selected_adapters=[name] if name else None
                    )
                    # Move files from subdirectory to save_dir for simpler loading
                    subdir = os.path.join(save_dir, name) if name else None
                    logger.info(f"Checking for adapter files in subdir: {subdir}")
                    if subdir and os.path.exists(subdir):
                        for f in os.listdir(subdir):
                            src = os.path.join(subdir, f)
                            dst = os.path.join(save_dir, f)
                            logger.info(f"Moving {src} -> {dst}")
                            shutil.move(src, dst)
                        os.rmdir(subdir)
                    logger.info(
                        f"Adapter saved. Files in {save_dir}: {os.listdir(save_dir)}"
                    )
            else:
                self.model.save_pretrained(save_dir, state_dict=state_dict)
                logger.info(
                    f"Full model saved. Files in {save_dir}: {os.listdir(save_dir)}"
                )

            try:
                self.tokenizer.save_pretrained(save_dir)
            except Exception:
                logger.warning("Failed to save tokenizer")

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
