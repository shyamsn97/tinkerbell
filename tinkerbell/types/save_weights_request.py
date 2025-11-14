from typing import Optional

from typing_extensions import Literal

from ._models import StrictBase
from .model_id import ModelID

__all__ = ["SaveWeightsRequest"]


class SaveWeightsRequest(StrictBase):
    """Request to save model weights."""

    model_id: ModelID
    """Model identifier"""

    path: Optional[str] = None
    """A file/directory name for the weights"""

    seq_id: Optional[int] = None
    """Optional sequence ID"""

    type: Literal["save_weights"] = "save_weights"
