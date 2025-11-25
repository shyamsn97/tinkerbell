from typing import Any

import numpy as np
import torch
from pydantic import Field, model_validator

from ._models import StrictBase
from .data import TensorData
from .loss_fn_inputs import LossFnInputs
from .model_input import ModelInput
from .tensor_dtype import _key_to_type

__all__ = ["Datum"]


class Datum(StrictBase):
    """Single data point combining model input and loss function inputs."""

    model_input: ModelInput
    """Model input as a ModelInput object or a list of token IDs"""

    loss_fn_inputs: LossFnInputs = Field(default_factory=dict)
    """Dictionary mapping field names to tensor data. Optional."""

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
        """Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction."""
        if isinstance(data, dict):
            # Only process loss_fn_inputs here - let ModelInput handle its own validation
            if "loss_fn_inputs" in data and isinstance(data["loss_fn_inputs"], dict):
                for inner_key, value in data["loss_fn_inputs"].items():
                    data["loss_fn_inputs"][inner_key] = cls._maybe_convert_array(
                        "loss_fn_inputs", value
                    )
        return data

    def to_torch(self, device: Any = None) -> dict[str, torch.Tensor]:
        """Convert Datum to a dictionary of torch tensors."""
        return {
            "model_input": self.model_input.to_torch(device=device),
            "loss_fn_inputs": {
                key: value.to_torch(device=device)
                for key, value in self.loss_fn_inputs.items()
            },
        }
