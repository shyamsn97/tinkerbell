from typing import Optional

from ._models import StrictBase

__all__ = ["LoraConfig", "SUPPORTED_LORA_TARGET_MODULES"]

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


class LoraConfig(StrictBase):
    rank: int
    """LoRA rank (dimension of low-rank matrices)"""

    seed: Optional[int] = None
    """Seed used for initialization of LoRA weights.

    Useful if you need deterministic or reproducible initialization of weights.
    """

    train_unembed: bool = False
    """Whether to add lora to the unembedding layer"""

    train_mlp: bool = True
    """Whether to add loras to the MLP layers (including MoE layers)"""

    train_attn: bool = True
    """Whether to add loras to the attention layers"""
