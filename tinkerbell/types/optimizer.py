from typing import Any

from pydantic import BaseModel

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


class ZeroGradRequest(BaseModel):
    model_name: str


class ZeroGradResponse(BaseModel):
    model_name: str
    message: str


class OptimStepRequest(BaseModel):
    model_name: str
    optimizer_params: dict[str, Any]


class OptimStepResponse(BaseModel):
    model_name: str
    message: str
