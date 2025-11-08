from typing import Any

from pydantic import BaseModel, Field

from tinkerbell.types.data import TensorData
from tinkerbell.types.optimizer import (
    DEFAULT_SCHEDULER_PARAMS,
)


class CreateTrainingActorsRequest(BaseModel):
    rank: int
    world_size: int
    master_addr: str
    master_port: str
    model_name: str
    model_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    parallelize_plan: dict[str, str] = Field(default_factory=lambda: {})
    scheduler_params: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SCHEDULER_PARAMS
    )
    lora_config: dict[str, Any] = Field(default_factory=lambda: {})
    ray_worker_options: dict[str, Any] = Field(default_factory=lambda: {})
    wait_until_ready: bool = False


class ForwardRequest(BaseModel):
    model_name: str
    request_id: str | None = None
    inputs: dict[str, TensorData] = Field(default_factory=lambda: {})
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    future: Any = None


class ForwardBackwardRequest(BaseModel):
    model_name: str
    request_id: str | None = None
    inputs: dict[str, TensorData] = Field(default_factory=lambda: {})
    targets: TensorData | None = None
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    return_logprobs: bool = False
    future: Any = None


class ActorStatusRequest(BaseModel):
    model_name: str
