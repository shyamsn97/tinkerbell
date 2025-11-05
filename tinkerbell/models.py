from typing import Any, List, Literal

from pydantic import BaseModel, Field

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


class CreateTrainingActorsRequest(BaseModel):
    rank: int
    world_size: int
    master_addr: str
    master_port: str
    model_name: str
    model_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    parallelize_plan: dict[str, str] = Field(default_factory=lambda: {})
    optimizer_params: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_OPTIMIZER_PARAMS
    )
    scheduler_params: dict[str, Any] = Field(
        default_factory=lambda: DEFAULT_SCHEDULER_PARAMS
    )
    lora_config: dict[str, Any] = Field(default_factory=lambda: {})
    ray_worker_options: dict[str, Any] = Field(default_factory=lambda: {})
    wait_until_ready: bool = False
    # metrics_fn: Callable
    # callbacks: list[Callable]


class CreateTrainingActorsResponse(BaseModel):
    success: bool
    model_name: str
    message: str


class RemoteFuture(BaseModel):
    request_id: str


class ForwardRequest(BaseModel):
    model_name: str
    request_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=lambda: {})
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    future: Any = None


class ForwardResponse(BaseModel):
    model_name: str
    request_id: str | None = None
    logprobs: List[List[int]] | None = None
    outputs: dict[str, Any] | None = None
    future: Any = None


class ForwardBackwardRequest(BaseModel):
    model_name: str
    request_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=lambda: {})
    targets: Any = None
    forward_kwargs: dict[str, Any] = Field(default_factory=lambda: {})
    return_logprobs: bool = False
    future: Any = None


class ForwardBackwardResponse(BaseModel):
    model_name: str
    request_id: str | None = None
    loss: float | None = None
    logprobs: List[List[int]] | None = None
    outputs: dict[str, Any] | None = None
    future: Any = None


class ActorStatusRequest(BaseModel):
    model_name: str


class ActorStatusResponse(BaseModel):
    status: Literal["ready", "initializing"]
    message: str | None = None


class DeployConfig(BaseModel):
    @property
    def deployment_type(self) -> str:
        return "local"


class ModalDeployConfig(DeployConfig):
    gpu: str = "H100"
    num_gpus: int = 1
    timeout: int = 86400
    serialized: bool = True
    container_idle_timeout: int = 600
    max_inputs: int = 100

    @property
    def deployment_type(self) -> str:
        return "modal"
