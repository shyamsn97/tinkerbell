from typing import Any, Literal

from pydantic import BaseModel

from tinkerbell.types.data import TensorData


class HealthResponse(BaseModel):
    status: str
    name: str


class CreateTrainingActorsResponse(BaseModel):
    success: bool
    model_name: str
    message: str


class ForwardResponse(BaseModel):
    model_name: str
    request_id: str | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, Any] | None = None
    future: Any = None


class ForwardBackwardResponse(BaseModel):
    model_name: str
    request_id: str | None = None
    loss: float | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, TensorData] | None = None
    future: Any = None


class ActorStatusResponse(BaseModel):
    status: Literal["ready", "initializing"]
    message: str | None = None


class RemoteFuture(BaseModel):
    request_id: str
