from tinkerbell.runtime.futures import Future, JobHandle
from tinkerbell.runtime.resources import ActorResources


def init(*args, **kwargs):
    from tinkerbell.runtime.session import init as _init

    return _init(*args, **kwargs)


def __getattr__(name: str):
    if name == "Session":
        from tinkerbell.runtime.session import Session

        return Session
    raise AttributeError(name)


__all__ = ["ActorResources", "Future", "JobHandle", "Session", "init"]
