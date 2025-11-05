from pydantic import BaseModel
from typing import Any, List, Literal

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
    logprobs: List[List[int]] | None = None
    outputs: dict[str, Any] | None = None
    future: Any = None

class ForwardBackwardResponse(BaseModel):
    model_name: str
    request_id: str | None = None
    loss: float | None = None
    logprobs: List[List[int]] | None = None
    outputs: dict[str, Any] | None = None
    future: Any = None

class ActorStatusResponse(BaseModel):
    status: Literal["ready", "initializing"]
    message: str | None = None
