"""Base Pydantic models and type aliases for tinkerbell types.

Adapted from tinker repository to maintain strict validation for requests
and flexible validation for responses.
"""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from typing_extensions import Literal, TypeAlias

__all__ = ["StrictBase", "BaseModel", "TensorDtype", "LossFnType", "_key_to_type"]

# Type aliases for tensor dtypes and loss functions
TensorDtype: TypeAlias = Literal["int64", "float32"]
LossFnType: TypeAlias = Literal["cross_entropy", "importance_sampling", "ppo"]

# Mapping from field names to expected tensor dtypes
_key_to_type = {
    # Integer types (token IDs, indices, masks)
    "input_ids": "int64",
    "attention_mask": "int64",
    "labels": "int64",
    "target_tokens": "int64",
    # Float types (probabilities, weights, thresholds)
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "ref_logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}


class StrictBase(PydanticBaseModel):
    """Base model for request types with strict validation.

    Don't allow extra fields, so user errors are caught earlier.
    Use this for request types.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)


class BaseModel(PydanticBaseModel):
    """Base model for response types with flexible validation.

    Use for classes that may appear in responses. Allow extra fields,
    so old clients can still work.
    """

    # For future-proofing, we ignore extra fields in case the server adds new fields.
    model_config = ConfigDict(frozen=True, extra="ignore")

    def __str__(self) -> str:
        return repr(self)
