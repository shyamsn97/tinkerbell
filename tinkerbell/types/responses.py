from typing import Any, Literal, Optional

from tinker.types import TensorData

from .base import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "healthy"]
    name: Optional[str] = None


class ZeroGradResponse(BaseModel):
    model_name: str
    message: str


class OptimStepResponse(BaseModel):
    model_name: str
    message: str
    metrics: Optional[dict[str, float]] = None


class CreateTrainingActorsResponse(BaseModel):
    success: bool
    model_name: str  # Actor group name
    message: str


class ForwardResponse(BaseModel):
    model_name: str  # Actor group name
    logprobs: TensorData | None = None
    outputs: dict[str, Any] | None = None
    metrics: Optional[dict[str, float]] = None


class ForwardBackwardResponse(BaseModel):
    model_name: str  # Actor group name
    loss: float | list[float] | None = None
    logprobs: TensorData | None = None
    outputs: dict[str, TensorData] | None = None
    metrics: Optional[dict[str, float]] = None
    sum_gradient: Optional[dict[str, float]] = None


class ActorStatusResponse(BaseModel):
    status: Literal["ready", "pending", "not_present"]
    message: str | None = None


class SaveCheckpointResponse(BaseModel):
    model_name: str  # Actor group name
    success: bool
    message: str | None = None
    path: Optional[str] = None


class PushToHubResponse(BaseModel):
    model_name: str  # Actor group name
    success: bool
    message: str | None = None
    repo_id: Optional[str] = None


class CreateSamplingActorResponse(BaseModel):
    success: bool
    message: str | None = None


class LogprobsResponse(BaseModel):
    logprobs: list[list[float]] | TensorData | None = None
    token_ids: list[list[int]] | TensorData | None = None


class SampleResponse(BaseModel):
    output: str
    tokens_generated: int | None = None
    # Logprobs for each token in the output
    logprobs: LogprobsResponse | None = None
    # Token IDs for the generated text
    output_token_ids: list[int] | None = None
    # Finish reason for the sequence
    finish_reason: str | None = None
    # Metadata about the sampling process
    meta_info: dict[str, Any] | None = None


class LoadCheckpointResponse(BaseModel):
    model_name: str  # Actor group name
    success: bool
    message: str | None = None
    lora_name: str | None = (
        None  # Actual lora_name used by SGLang (may differ from checkpoint_path)
    )


class ShutdownSamplingActorResponse(BaseModel):
    model_name: str  # Actor group name
    success: bool
    message: str | None = None


class PollResultResponse(BaseModel):
    status: Literal["pending", "completed", "error"]
    request_id: str
    result: dict[str, Any] | None = None
    error: str | None = None
