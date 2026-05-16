"""Base Pydantic models and type aliases.

Re-exports core types from tinker SDK for API compatibility.
"""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from tinker.types.loss_fn_type import LossFnType
from tinker.types.tensor_dtype import TensorDtype

__all__ = ["StrictBase", "BaseModel", "TensorDtype", "LossFnType", "_key_to_type"]

# Mapping from field names to expected tensor dtypes (used internally for auto-conversion)
_key_to_type = {
    "input_ids": "int64",
    "attention_mask": "int64",
    "labels": "int64",
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "sampling_logprobs": "float32",
    "ref_logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}


class StrictBase(PydanticBaseModel):
    """Base model for request types with strict validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    def __str__(self) -> str:
        return repr(self)


class BaseModel(PydanticBaseModel):
    """Base model for response types with flexible validation."""

    model_config = ConfigDict(frozen=True, extra="ignore", arbitrary_types_allowed=True)

    def __str__(self) -> str:
        return repr(self)
