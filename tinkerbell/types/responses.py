from typing import Any, Literal, Optional

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


class ForwardResponse(BaseModel):
    model_name: str  # Actor group name
    logprobs: Any = None
    outputs: dict[str, Any] | None = None
    metrics: Optional[dict[str, float]] = None


class ForwardBackwardResponse(BaseModel):
    model_name: str  # Actor group name
    loss: float | list[float] | None = None
    logprobs: Any = None
    outputs: dict[str, Any] | None = None
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


class LogprobsResponse(BaseModel):
    logprobs: Any = None
    token_ids: Any = None


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
