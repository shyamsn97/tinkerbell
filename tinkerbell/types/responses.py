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


class ForwardBackwardResponse(BaseModel):
    model_name: str
    request_id: str | None = None
    loss: float | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, TensorData] | None = None


class ActorStatusResponse(BaseModel):
    status: Literal["ready", "initializing"]
    message: str | None = None


class RemoteFuture(BaseModel):
    request_id: str


class GetRayActorsResponse(BaseModel):
    actor_names: list[str]


class SaveCheckpointResponse(BaseModel):
    model_name: str
    success: bool
    message: str | None = None


class CreateInferenceActorResponse(BaseModel):
    success: bool
    message: str | None = None


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int | None = None
