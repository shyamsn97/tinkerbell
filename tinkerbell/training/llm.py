import logging
import os
import re
import shutil
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
        enable_gradient_checkpointing: bool = True,
    ):
        self.rank = rank
        self.world_size = world_size
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.parallelize_plan = parallelize_plan
        self.lora_config = lora_config
        self.adapter_name = adapter_name or "default"
        self.initialize_random_weights = initialize_random_weights
        self.should_merge_lora = {}
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.adapters: dict[str, LoraConfig] = {}
        self.active_adapter: str | None = None
        self.tokenizer = None
        self.setup()

    def setup(self) -> None:
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

        # Enable gradient checkpointing BEFORE PEFT wrapping (required for PEFT compatibility)
        # See: https://github.com/huggingface/peft/issues/2826
        if self.enable_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info(
                    "Gradient checkpointing enabled on base model (before PEFT)"
                )
            elif hasattr(self.model, "config") and hasattr(
                self.model.config, "gradient_checkpointing"
            ):
                self.model.config.gradient_checkpointing = True
                logger.info("Gradient checkpointing enabled via config (before PEFT)")

        if self.lora_config is not None:
            self.model = self.add_adapter(self.adapter_name, self.lora_config)

        self.model = self.parallelize(
            model=self.model, parallelize_plan=self.parallelize_plan
        )

    def train(self) -> None:
        self.model.train()

    def create_model(self, model_id: str, model_kwargs: dict[str, Any]) -> nn.Module:
        from transformers import AutoConfig, AutoModelForCausalLM

        # Remove gradient_checkpointing from kwargs - it's not a valid model init argument
        # We'll enable it separately after model creation
        filtered_kwargs = {
            k: v for k, v in model_kwargs.items() if k != "gradient_checkpointing"
        }

        config = AutoConfig.from_pretrained(model_id)
        if self.initialize_random_weights:
            return AutoModelForCausalLM.from_config(config, **filtered_kwargs)
        return AutoModelForCausalLM.from_pretrained(self.model_id, **filtered_kwargs)

    def _build_target_modules(self, lora_config: LoraConfig) -> list[str]:
        target_modules = []
        if lora_config.train_attn:
            target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"])
        if lora_config.train_mlp:
            target_modules.extend(["gate_proj", "up_proj", "down_proj", "gate_up_proj"])
        if lora_config.train_unembed:
            target_modules.extend(["lm_head", "embed_out"])
        return target_modules

    def add_adapter(self, adapter_name: str, lora_config: LoraConfig) -> nn.Module:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        if adapter_name in self.adapters:
            self.set_active_adapter(adapter_name)
            return self.model

        target_modules = self._build_target_modules(lora_config)
        peft_config = PeftLoraConfig(
            r=lora_config.rank,
            lora_alpha=lora_config.rank * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if not self.adapters:
            self.model = get_peft_model(
                self.model, peft_config, adapter_name=adapter_name
            )
        else:
            self.model.add_adapter(adapter_name, peft_config)

        self.adapters[adapter_name] = lora_config
        self.set_active_adapter(adapter_name)

        unsupported = [
            tm
            for tm in target_modules
            if not any(
                re.search(re.escape(s), tm) for s in SUPPORTED_LORA_TARGET_MODULES
            )
        ]
        self.should_merge_lora[adapter_name] = bool(unsupported)

        return self.model

    def set_active_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.adapters:
            raise ValueError(
                f"Adapter '{adapter_name}' not found. Available: {list(self.adapters.keys())}"
            )
        self.model.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def pad(self, data: list[Datum], device: torch.device) -> dict[str, torch.Tensor]:
        from tinkerbell.utils import get_nested, set_nested

        torch_data = [d.to_torch(device=device) for d in data]
        result = {}

        for path, padding_strategy in self.padding_strategies.items():
            values = [get_nested(d, path) for d in torch_data]
            if all(v is None for v in values):
                continue

            if any(v is None for v in values) and "attention_mask" in path:
                input_ids_path = path.replace("attention_mask", "input_ids")
                input_ids_values = [get_nested(d, input_ids_path) for d in torch_data]
                values = [
                    torch.ones_like(input_ids_values[i]) if v is None else v
                    for i, v in enumerate(values)
                ]
            elif any(v is None for v in values):
                continue

            set_nested(result, path, padding_strategy.pad_sequence(values))
        return result

    def forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        with_grad: bool = True,
        forward_kwargs: dict[str, Any] = {},
    ) -> torch.Tensor:
        try:
            if with_grad:
                self.model.train()
                outputs = self.model(**model_inputs, **forward_kwargs)
            else:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(**model_inputs, **forward_kwargs)
            # Extract logits and explicitly delete outputs to free memory
            logits = outputs.logits
            del outputs
            return logits
        except Exception as e:
            logger.error(f"Forward error: {e}")
            raise

    def get_model_state_dict(self, full_state_dict: bool = False):
        options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=True)
        return get_model_state_dict(self.model, options=options)

    def save_model(self, save_dir: str, adapter_name: str | None = None):
        state_dict = None
        if not self.adapters:
            state_dict = self.get_model_state_dict(full_state_dict=True)

        if self.rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            if self.adapters:
                name = adapter_name or self.active_adapter
                if name and self.should_merge_lora.get(name, False):
                    self.model = self.model.merge_and_unload()
                    merged_state = self.get_model_state_dict(full_state_dict=True)
                    self.model.save_pretrained(save_dir, state_dict=merged_state)
                else:
                    self.model.save_pretrained(
                        save_dir, selected_adapters=[name] if name else None
                    )
                    subdir = os.path.join(save_dir, name) if name else None
                    if subdir and os.path.exists(subdir):
                        for f in os.listdir(subdir):
                            shutil.move(
                                os.path.join(subdir, f), os.path.join(save_dir, f)
                            )
                        os.rmdir(subdir)
            else:
                self.model.save_pretrained(save_dir, state_dict=state_dict)
            try:
                self.tokenizer.save_pretrained(save_dir)
            except Exception:
                pass

        dist.barrier()

    def parallelize(
        self, model: nn.Module, parallelize_plan: dict[str, str]
    ) -> nn.Module:
        if not parallelize_plan:
            return model.cuda()

        strategies = {"column": ColwiseParallel, "row": RowwiseParallel}
        module_parallelization_plan = {}
        for pattern, strategy_name in parallelize_plan.items():
            strategy = strategies[strategy_name]()
            for name in get_submodules_with_wildcard(model, pattern):
                module_parallelization_plan[name] = strategy

        device_mesh = init_device_mesh(
            "cuda", (1, self.world_size), mesh_dim_names=("dp", "tp")
        )
        model = parallelize_module(
            model, device_mesh["tp"], module_parallelization_plan
        )
        return model.cuda()
