from typing import Any, Literal, Optional

from ._models import BaseModel
from .data import TensorData
from .model_id import ModelID
from .request_id import RequestID


class HealthResponse(BaseModel):
    status: Literal["ok", "healthy"]
    name: Optional[str] = None


class CreateTrainingActorsResponse(BaseModel):
    success: bool
    model_id: ModelID
    message: str


class ForwardResponse(BaseModel):
    model_id: ModelID
    request_id: RequestID | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, Any] | None = None
    metrics: Optional[dict[str, float]] = None


class ForwardBackwardResponse(BaseModel):
    model_id: ModelID
    request_id: RequestID | None = None
    loss: float | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, TensorData] | None = None
    metrics: Optional[dict[str, float]] = None


class ActorStatusResponse(BaseModel):
    status: Literal["ready", "initializing", "not_present"]
    message: str | None = None


class RemoteFuture(BaseModel):
    request_id: RequestID


class GetRayActorsResponse(BaseModel):
    actor_names: list[str]


class SaveCheckpointResponse(BaseModel):
    model_id: ModelID
    success: bool
    message: str | None = None
    path: Optional[str] = None


class CreateInferenceActorResponse(BaseModel):
    success: bool
    message: str | None = None


class GenerateResponse(BaseModel):
    outputs: list[str]
    tokens_generated: int | None = None
