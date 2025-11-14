from typing import Any, Optional

from pydantic import Field

from ._models import StrictBase
from .data import TensorData
from .lora_config import LoraConfig
from .model_id import ModelID
from .optimizer import DEFAULT_SCHEDULER_PARAMS
from .request_id import RequestID


class CreateTrainingActorsRequest(StrictBase):
    world_size: int
    model_id: ModelID
    model_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    parallelize_plan: dict[str, str] = Field(default_factory=lambda: {})
    scheduler_params: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SCHEDULER_PARAMS
    )
    lora_config: Optional[LoraConfig | dict[str, Any]] = Field(default=None)
    ray_worker_options: dict[str, Any] = Field(default_factory=lambda: {})
    from_pretrained: bool = True
    wait_until_ready: bool = False


class SaveCheckpointRequest(StrictBase):
    model_id: ModelID
    checkpoint_path: str
    path: Optional[str] = None
    seq_id: Optional[int] = None


class ActorRequest(StrictBase):
    request_id: RequestID


class ForwardRequest(StrictBase):
    model_id: ModelID
    request_id: Optional[RequestID] = None
    seq_id: Optional[int] = None
    inputs: dict[str, TensorData] = Field(default_factory=lambda: {})
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    future: Any = None


class ForwardBackwardRequest(StrictBase):
    model_id: ModelID
    request_id: Optional[RequestID] = None
    seq_id: Optional[int] = None
    inputs: dict[str, TensorData] = Field(default_factory=lambda: {})
    targets: TensorData | None = None
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    return_logprobs: bool = False
    future: Any = None


class ActorStatusRequest(StrictBase):
    model_id: ModelID


class CreateInferenceActorRequest(StrictBase):
    model_id: ModelID
    tp_size: int
    engine_kwargs: dict[str, Any] = Field(default_factory=lambda: {})


class GenerateRequest(StrictBase):
    model_id: ModelID
    prompts: list[str]
    max_tokens: int = 100
    temperature: float = 0.7
