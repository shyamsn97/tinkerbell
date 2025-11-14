from typing import Optional

from typing_extensions import Literal

from ._models import StrictBase
from .model_id import ModelID

__all__ = ["LoadWeightsRequest"]


class LoadWeightsRequest(StrictBase):
    """Request to load model weights."""

    model_id: ModelID
    """Model identifier"""

    path: str
    """A URI for model weights at a specific step"""

    seq_id: Optional[int] = None
    """Optional sequence ID"""

    type: Literal["load_weights"] = "load_weights"
