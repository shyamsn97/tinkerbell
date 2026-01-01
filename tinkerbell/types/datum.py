from typing import Any

import torch
from pydantic import Field, model_validator

from ._models import StrictBase
from .data import TensorData
from .model_input import ModelInput

__all__ = ["Datum"]


class Datum(StrictBase):
    """Single data point combining model input and loss function inputs."""

    model_input: ModelInput
    """Model input as a ModelInput object or a list of token IDs"""

    loss_fn_inputs: dict[str, TensorData] = Field(default_factory=dict)
    """Dictionary mapping field names to tensor data. Optional."""

    def get_input_ids(self) -> TensorData:
        return self.model_input.input_ids

    @model_validator(mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        """Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction."""
        from tinkerbell.utils import convert_to_tensor_data, process_dict_values

        if isinstance(data, dict):
            if "loss_fn_inputs" in data and isinstance(data["loss_fn_inputs"], dict):
                data["loss_fn_inputs"] = process_dict_values(
                    data["loss_fn_inputs"], convert_to_tensor_data
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
