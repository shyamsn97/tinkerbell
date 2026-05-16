from tinkerbell.runtime.futures import Future, JobHandle
from tinkerbell.runtime.resources import ActorResources


def init(*args, **kwargs):
    from tinkerbell.runtime.session import init as _init

    return _init(*args, **kwargs)


def deploy_service(*args, **kwargs):
    from tinkerbell.api.server import deploy_service as _deploy_service

    return _deploy_service(*args, **kwargs)


def __getattr__(name: str):
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
    "Future",
    "JobHandle",
    "Sampler",
    "Session",
    "TinkerbellServer",
    "TrainGroup",
    "deploy_service",
    "init",
]
