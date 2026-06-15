from tinkerbell.runtime.futures import Future, JobHandle
from tinkerbell.runtime.resources import ActorResources


def init(*args, **kwargs):
    from tinkerbell.runtime.session import init as _init

    return _init(*args, **kwargs)


def deploy_service(*args, **kwargs):
    from tinkerbell.api.server import deploy_service as _deploy_service

    return _deploy_service(*args, **kwargs)


def __getattr__(name: str):
    if name == "ServiceClient":
        from tinkerbell.client import ServiceClient

        return ServiceClient
    if name in {
        "AdamParams",
        "Datum",
        "EncodedTextChunk",
        "LoraConfig",
        "ModelInput",
        "SamplingParams",
        "TensorData",
    }:
        from tinkerbell import types

        return getattr(types, name)
    if name == "Session":
        from tinkerbell.runtime.session import Session

        return Session
    if name == "TrainGroup":
        from tinkerbell.training import TrainGroup

        return TrainGroup
    if name == "Sampler":
        from tinkerbell.sampling import Sampler

        return Sampler
    if name == "TinkerbellServer":
        from tinkerbell.api.server import TinkerbellServer

        return TinkerbellServer
    raise AttributeError(name)


__all__ = [
    "ActorResources",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "Future",
    "JobHandle",
    "LoraConfig",
    "ModelInput",
    "SamplingParams",
    "Sampler",
    "Session",
    "ServiceClient",
    "TensorData",
    "TinkerbellServer",
    "TrainGroup",
    "deploy_service",
    "init",
]
