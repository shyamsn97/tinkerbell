"""Tinkerbell types - re-exports tinker SDK types for public API compatibility,
plus internal types for server/actor protocol."""

# Core types from tinker SDK (public API)
from tinker.types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LossFnType,
    LoraConfig,
    ModelInput,
    SampledSequence,
    SamplingParams,
    TensorData,
    TensorDtype,
)

# Re-export SampleResponse from tinker as TinkerSampleResponse to avoid clash
from tinker.types import SampleResponse as TinkerSampleResponse

# Internal base models
from ._models import BaseModel, StrictBase

# Internal data utilities
from .data import PaddingStrategy

# Infrastructure types (tinkerbell-specific)
from .deploy import DeployConfig, ModalDeployConfig

# Internal server protocol types
from .optimizer import (
    OptimStepRequest,
    OptimStepResponse,
    ZeroGradRequest,
    ZeroGradResponse,
)
from .requests import (
    ActorStatusRequest,
    CreateSamplingActorRequest,
    CreateTrainingActorsRequest,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadCheckpointRequest,
    PollResultRequest,
    PushToHubRequest,
    SampleRequest,
    SaveCheckpointRequest,
    ShutdownSamplingActorRequest,
)
from .responses import (
    ActorStatusResponse,
    CreateSamplingActorResponse,
    CreateTrainingActorsResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    GetRayActorsResponse,
    HealthResponse,
    LoadCheckpointResponse,
    PollResultResponse,
    PushToHubResponse,
    RemoteFuture,
    SaveCheckpointResponse,
    ShutdownSamplingActorResponse,
)

# Internal SampleResponse (different from tinker's - this is our SGLang wrapper)
from .responses import SampleResponse as InternalSampleResponse

__all__ = [
    # tinker SDK types (public API)
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
    "TinkerSampleResponse",
    # Internal base models
    "BaseModel",
    "StrictBase",
    # Internal data utilities
    "PaddingStrategy",
    # Infrastructure
    "DeployConfig",
    "ModalDeployConfig",
    # Internal server protocol
    "OptimStepRequest",
    "OptimStepResponse",
    "ZeroGradRequest",
    "ZeroGradResponse",
    "ActorStatusRequest",
    "CreateSamplingActorRequest",
    "CreateTrainingActorsRequest",
    "ForwardBackwardRequest",
    "ForwardRequest",
    "LoadCheckpointRequest",
    "PollResultRequest",
    "PushToHubRequest",
    "SampleRequest",
    "SaveCheckpointRequest",
    "ShutdownSamplingActorRequest",
    "ActorStatusResponse",
    "CreateSamplingActorResponse",
    "CreateTrainingActorsResponse",
    "ForwardBackwardResponse",
    "ForwardResponse",
    "GetRayActorsResponse",
    "HealthResponse",
    "InternalSampleResponse",
    "LoadCheckpointResponse",
    "PollResultResponse",
    "PushToHubResponse",
    "RemoteFuture",
    "SaveCheckpointResponse",
    "ShutdownSamplingActorResponse",
]
