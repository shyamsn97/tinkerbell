from typing import Any, Literal, Optional

from ._models import BaseModel
from .data import TensorData


class HealthResponse(BaseModel):
    status: Literal["ok", "healthy"]
    name: Optional[str] = None


class CreateTrainingActorsResponse(BaseModel):
    success: bool
    model_id: str
    message: str


class ForwardResponse(BaseModel):
    model_id: str
    request_id: str | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, Any] | None = None
    metrics: Optional[dict[str, float]] = None


class ForwardBackwardResponse(BaseModel):
    model_id: str
    request_id: str | None = None
    loss: float | list[float] | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, TensorData] | None = None
    metrics: Optional[dict[str, float]] = None
    sum_gradient: Optional[dict[str, float]] = None


class ActorStatusResponse(BaseModel):
    status: Literal["ready", "pending", "not_present"]
    message: str | None = None


class RemoteFuture(BaseModel):
    request_id: str
    model_id: str | None = None


class GetRayActorsResponse(BaseModel):
    actor_names: list[str]


class SaveCheckpointResponse(BaseModel):
    model_id: str
    success: bool
    message: str | None = None
    path: Optional[str] = None


class CreateSamplingActorResponse(BaseModel):
    success: bool
    message: str | None = None


class SampleResponse(BaseModel):
    outputs: list[str]
    tokens_generated: int | None = None
    # Logprobs for each token in the output. Can be logits if logprobs not available from backend
    logprobs: list[list[float]] | None = None
    # Top k logprobs for each position (if requested)
    top_logprobs: list[dict[str, float]] | None = None
    # Token IDs for the generated text
    output_token_ids: list[list[int]] | None = None
    # Finish reasons for each sequence
    finish_reasons: list[str] | None = None
    # Metadata about the sampling process
    meta_info: dict[str, Any] | None = None


class LoadCheckpointResponse(BaseModel):
    model_id: str
    success: bool
    message: str | None = None


class ShutdownSamplingActorResponse(BaseModel):
    model_id: str
    success: bool
    message: str | None = None


class PollResultResponse(BaseModel):
    status: Literal["pending", "completed", "error"]
    request_id: str
    result: dict[str, Any] | None = None
    error: str | None = None
