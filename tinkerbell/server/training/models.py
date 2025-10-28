from pydantic import BaseModel, Field, Callable
from typing import Any
import torch

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

class HealthResponse(BaseModel):
    status: str
    name: str

class SetupTrainRequest(BaseModel):
    rank: int
    world_size: int
    master_addr: str
    master_port: str
    model_path: str
    parallelize_plan: dict[str, str]
    optimizer_params: dict[str, Any] = Field(default_factory=lambda: DEFAULT_OPTIMIZER_PARAMS)
    scheduler_params: dict[str, Any] = Field(default_factory=lambda: DEFAULT_SCHEDULER_PARAMS)
    # metrics_fn: Callable
    # callbacks: list[Callable]

class SetupTrainResponse(BaseModel):
    success: bool
    message: str

class ForwardRequest(BaseModel):
    inputs: dict[str, torch.Tensor]
    forward_kwargs: dict[str, Any] = Field(default_factory=dict)

class ForwardResponse(BaseModel):
    outputs: dict[str, torch.Tensor]

class ForwardBackwardRequest(BaseModel):
    inputs: dict[str, torch.Tensor]
    forward_kwargs: dict[str, Any] = Field(default_factory=dict)

class ForwardBackwardResponse(BaseModel):
    loss: float | None