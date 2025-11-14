from typing_extensions import Literal

from ._models import BaseModel
from .model_id import ModelID

__all__ = ["CreateModelResponse"]


class CreateModelResponse(BaseModel):
    """Response from creating a model."""

    model_id: ModelID
    """Unique identifier for the created model"""

    type: Literal["create_model"] = "create_model"
