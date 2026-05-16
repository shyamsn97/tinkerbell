"""Tiny Future wrapper over Ray ObjectRefs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Future(Generic[T]):
    """A small sync/async handle around one or more Ray ObjectRefs."""

    def __init__(
        self,
        refs,
        combine: Callable[[list], T] | None = None,
    ):
        self.refs = refs if isinstance(refs, list) else [refs]
        self.combine = combine or (
            lambda values: values[0] if len(values) == 1 else values
        )
        self._resolved = False
        self._value: T | None = None

    @property
    def done(self) -> bool:
        import ray

        if self._resolved:
            return True
        ready, _ = ray.wait(self.refs, num_returns=len(self.refs), timeout=0)
        return len(ready) == len(self.refs)

    def result(self) -> T:
        import ray

        if not self._resolved:
            self._value = self.combine(ray.get(self.refs))
            self._resolved = True
        return self._value  # type: ignore[return-value]

    def __await__(self):
        async def wait() -> T:
            return await asyncio.to_thread(self.result)

        return wait().__await__()


JobHandle = Future

__all__ = ["Future", "JobHandle"]
