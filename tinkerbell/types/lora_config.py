"""LoraConfig type re-exported from tinker SDK."""

from tinker.types.lora_config import LoraConfig

__all__ = ["LoraConfig"]

SUPPORTED_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
