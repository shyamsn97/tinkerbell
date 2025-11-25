from __future__ import annotations

from typing import Any, Dict, List

import torch
from pydantic import model_validator

from ._models import StrictBase
from .data import TensorData

__all__ = ["ModelInput"]


class ModelInput(StrictBase):
    """Model input representing token sequences and associated data.

    Simplified from tinker's chunk-based approach for tinkerbell compatibility.
    """

    tokens: TensorData | List[int] | Any
    """List of input token IDs"""

    attention_mask: TensorData | List[int] | Any | None = None
    """Optional attention mask"""

    additional_inputs: Dict[str, TensorData | List[int] | Any] | None = None
    """Optional additional inputs as tensors"""

    @model_validator(mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and lists to TensorData during construction."""
        from tinkerbell.utils import convert_to_tensor_data, process_dict_values

        if isinstance(data, dict):
            if "tokens" in data and data["tokens"] is not None:
                data["tokens"] = convert_to_tensor_data("tokens", data["tokens"])

            if "attention_mask" in data and data["attention_mask"] is not None:
                data["attention_mask"] = convert_to_tensor_data(
                    "attention_mask", data["attention_mask"]
                )

            if "additional_inputs" in data and isinstance(
                data["additional_inputs"], dict
            ):
                data["additional_inputs"] = process_dict_values(
                    data["additional_inputs"], convert_to_tensor_data
                )

            if "labels" in data and data["labels"] is not None:
                data["labels"] = convert_to_tensor_data("labels", data["labels"])

        return data

    def __len__(self) -> int:
        """Return the total context length."""
        return len(self.tokens)

    def to_torch(self, device: str = "cuda") -> dict[str, torch.Tensor]:
        """Convert ModelInput to a dictionary of torch tensors."""
        return {
            "tokens": self.tokens.to_torch(device=device),
            "attention_mask": (
                self.attention_mask.to_torch(device=device)
                if self.attention_mask is not None
                else None
            ),
            "additional_inputs": {
                key: value.to_torch(device=device)
                for key, value in self.additional_inputs.items()
                if value is not None
            },
        }
