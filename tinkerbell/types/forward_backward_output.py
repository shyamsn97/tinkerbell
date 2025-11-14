from typing import Dict, List

from ._models import BaseModel
from .loss_fn_output import LossFnOutput

__all__ = ["ForwardBackwardOutput"]


class ForwardBackwardOutput(BaseModel):
    """Output from forward/backward pass with metrics."""

    loss_fn_output_type: str
    """The type of the ForwardBackward output"""

    loss_fn_outputs: List[LossFnOutput]
    """List of loss function outputs (tensor dictionaries)"""

    metrics: Dict[str, float]
    """Training metrics as key-value pairs"""
