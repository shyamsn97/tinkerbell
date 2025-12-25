from typing import Any, Optional

from ._models import BaseModel, StrictBase

DEFAULT_OPTIMIZER_PARAMS: dict[str, Any] = {
    "name": "adamw",
    "lr": 5e-5,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.01,
}
DEFAULT_SCHEDULER_PARAMS: dict[str, Any] = {"scheduler": "cosine"}


class ZeroGradRequest(StrictBase):
    model_name: str
    adapter_name: Optional[str] = None


class ZeroGradResponse(BaseModel):
    model_name: str
    message: str


class OptimStepRequest(StrictBase):
    model_name: str
    request_id: Optional[str] = None
    adapter_name: Optional[str] = None
    optimizer_params: dict[str, Any] = {}
    immediate: bool = False  # If True, process the batch queue immediately


class OptimStepResponse(BaseModel):
    model_name: str
    message: str
    metrics: Optional[dict[str, float]] = None
