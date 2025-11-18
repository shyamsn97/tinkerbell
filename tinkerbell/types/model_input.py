from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from pydantic import model_validator

from ._models import StrictBase
from .data import TensorData
from .tensor_dtype import _key_to_type

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

    @classmethod
    def _maybe_convert_array(cls, key: str, value: Any) -> Any:
        """Convert torch.Tensor, numpy array, dict, or 1-D list to TensorData if needed."""
        if isinstance(value, TensorData):
            # Already a TensorData, no conversion needed
            return value
        elif isinstance(value, torch.Tensor):
            return TensorData.from_torch(value)
        elif isinstance(value, np.ndarray):
            return TensorData.from_numpy(value)
        elif (
            isinstance(value, dict)
            and "data" in value
            and "dtype" in value
            and "shape" in value
        ):
            # Reconstruct TensorData from serialized dict (from JSON/model_dump)
            return TensorData(**value)
        elif isinstance(value, list):
            # assume it's 1d and infer the dtype from the key
            return TensorData(
                data=value, dtype=_key_to_type.get(key, "float32"), shape=[len(value)]
            )
        else:
            return value

    @model_validator(mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and lists to TensorData during construction."""
        if isinstance(data, dict):
            # Handle tokens
            if "tokens" in data and data["tokens"] is not None:
                data["tokens"] = cls._maybe_convert_array("tokens", data["tokens"])

            # Handle attention_mask
            if "attention_mask" in data and data["attention_mask"] is not None:
                data["attention_mask"] = cls._maybe_convert_array(
                    "attention_mask", data["attention_mask"]
                )

            # Handle additional_inputs - this is a dict of values that need conversion
            if "additional_inputs" in data and isinstance(
                data["additional_inputs"], dict
            ):
                for inner_key, value in data["additional_inputs"].items():
                    data["additional_inputs"][inner_key] = cls._maybe_convert_array(
                        inner_key, value
                    )
        return data

    @classmethod
    def from_tokens(cls, tokens: List[int] | torch.Tensor) -> ModelInput:
        """Create a ModelInput from a list of token IDs."""
        if isinstance(tokens, torch.Tensor):
            token_data = TensorData.from_torch(tokens)
        else:
            token_data = TensorData(data=tokens, dtype="int64", shape=[len(tokens)])
        return cls(tokens=token_data)

    @property
    def length(self) -> int:
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
