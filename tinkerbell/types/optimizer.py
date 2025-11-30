from typing import Any, Optional

from typing_extensions import Literal

from ._models import BaseModel, StrictBase

DEFAULT_OPTIMIZER_PARAMS: dict[str, Any] = {
    "name": "adamw",
    "lr": 5e-5,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.01,
}
DEFAULT_SCHEDULER_PARAMS: dict[str, Any] = {
    "scheduler": "cosine",
}


class AdamParams(StrictBase):
    """Adam optimizer parameters."""

    learning_rate: float = 0.0001
    """Learning rate for the optimizer"""

    beta1: float = 0.9
    """Coefficient used for computing running averages of gradient"""

    beta2: float = 0.95
    """Coefficient used for computing running averages of gradient square"""

    eps: float = 1e-12
    """Term added to the denominator to improve numerical stability"""


class ZeroGradRequest(StrictBase):
    model_id: str  # This is model_name for routing
    adapter_name: Optional[str] = None  # Which LoRA adapter


class ZeroGradResponse(BaseModel):
    model_id: str
    message: str


class OptimStepRequest(StrictBase):
    model_id: str  # This is model_name for routing
    adapter_name: Optional[str] = None  # Which LoRA adapter
    optimizer_params: dict[str, Any]
    adam_params: Optional[AdamParams] = None
    seq_id: Optional[int] = None
    type: Optional[Literal["optim_step"]] = None


class OptimStepResponse(BaseModel):
    model_id: str
    message: str
    metrics: Optional[dict[str, float]] = None
