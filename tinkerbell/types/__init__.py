"""Tinkerbell types.

Public imports re-export tinker SDK types for API compatibility. Internal
types cover the HTTP protocol and the server-internal ops/jobs pipeline.
"""

from tinker.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LoraConfig,
    LossFnType,
    ModelInput,
    SampledSequence,
    SamplingParams,
    TensorData,
    TensorDtype,
)

from tinkerbell.types.base import BaseModel, StrictBase
from tinkerbell.types.data import PaddingStrategy
from tinkerbell.types.deploy import DeployConfig, ModalDeployConfig
from tinkerbell.types.jobs import (
    ErrorRecord,
    JobHandle,
    JobKind,
    JobRecord,
    JobStatus,
    PollRequest,
    SubmitResponse,
)
from tinkerbell.types.requests import (
    ActorStatusRequest,
    CreateSamplingActorRequest,
    CreateTrainingActorsRequest,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadCheckpointRequest,
    OptimStepRequest,
    PollResultRequest,
    PushToHubRequest,
    SampleRequest,
    SaveCheckpointRequest,
    ShutdownSamplingActorRequest,
    ZeroGradRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    CreateSamplingActorResponse,
    CreateTrainingActorsResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    HealthResponse,
    LoadCheckpointResponse,
    OptimStepResponse,
    PollResultResponse,
    PushToHubResponse,
    SampleResponse,
    SaveCheckpointResponse,
    ShutdownSamplingActorResponse,
    ZeroGradResponse,
)
from tinkerbell.types.route import RouteKey

__all__ = [
    # tinker SDK re-exports
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LossFnType",
    "LoraConfig",
    "ModelInput",
    "SampledSequence",
    "SamplingParams",
    "TensorData",
    "TensorDtype",
    # Base models
    "BaseModel",
    "StrictBase",
    # Data utilities
    "PaddingStrategy",
    # Infrastructure
    "DeployConfig",
    "ModalDeployConfig",
    # Job system
    "ErrorRecord",
    "JobHandle",
    "JobKind",
    "JobRecord",
    "JobStatus",
    "PollRequest",
    "SubmitResponse",
    # Routing
    "RouteKey",
    # Protocol — requests
    "ActorStatusRequest",
    "CreateSamplingActorRequest",
    "CreateTrainingActorsRequest",
    "ForwardBackwardRequest",
    "ForwardRequest",
    "LoadCheckpointRequest",
    "OptimStepRequest",
    "PollResultRequest",
    "PushToHubRequest",
    "SampleRequest",
    "SaveCheckpointRequest",
    "ShutdownSamplingActorRequest",
    "ZeroGradRequest",
    # Protocol — responses
    "ActorStatusResponse",
    "CreateSamplingActorResponse",
    "CreateTrainingActorsResponse",
    "ForwardBackwardResponse",
    "ForwardResponse",
    "HealthResponse",
    "LoadCheckpointResponse",
    "OptimStepResponse",
    "PollResultResponse",
    "PushToHubResponse",
    "SampleResponse",
    "SaveCheckpointResponse",
    "ShutdownSamplingActorResponse",
    "ZeroGradResponse",
]
