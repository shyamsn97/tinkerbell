from typing import Optional

from typing_extensions import Literal

from ._models import BaseModel

__all__ = ["LoadWeightsResponse"]


class LoadWeightsResponse(BaseModel):
    """Response from loading weights."""

    success: bool
    """Whether the load was successful"""

    message: Optional[str] = None
    """Optional message"""

    type: Optional[Literal["load_weights"]] = None
