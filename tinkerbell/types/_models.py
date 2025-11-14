"""Base Pydantic models for tinkerbell types.

Adapted from tinker repository to maintain strict validation for requests
and flexible validation for responses.
"""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

__all__ = ["StrictBase", "BaseModel"]


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
