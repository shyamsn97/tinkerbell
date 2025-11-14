from typing import Any, Dict, List, Optional

from ._models import StrictBase
from .loss_fn_type import LossFnType

__all__ = ["ForwardBackwardInput"]


class ForwardBackwardInput(StrictBase):
    """Input data for forward/backward pass.

    Adapted from tinker to work with tinkerbell's data structures.
    """

    data: List[Dict[str, Any]]
    """Array of input data for the forward/backward pass"""

    loss_fn: LossFnType
    """Fully qualified function path for the loss function"""

    loss_fn_config: Optional[Dict[str, float]] = None
    """Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)"""
