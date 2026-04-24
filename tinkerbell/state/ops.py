"""Server-internal Op types — the dispatchable units on WorkQueue.

These are the vocabulary of the data plane: what the gateway submits and
what the training executor pulls. They never cross HTTP, so they are
plain dataclasses rather than pydantic wire models. They live next to
`WorkQueue` / `Registry` because they are coupled to both: `WorkQueue`
holds lists of them, `TrainingExecutor` dispatches on their type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from tinker.types import Datum, LossFnType

from tinkerbell.types.route import RouteKey


@dataclass
class ForwardOp:
    job_id: str
    route: RouteKey
    data: list[Datum]
    forward_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardBackwardOp:
    job_id: str
    route: RouteKey
    data: list[Datum]
    forward_kwargs: dict[str, Any] = field(default_factory=dict)
    return_logprobs: bool = False
    loss_fn: LossFnType = "cross_entropy"
    loss_fn_config: dict[str, float] | None = None


@dataclass
class ZeroGradOp:
    job_id: str
    route: RouteKey


@dataclass
class OptimStepOp:
    job_id: str
    route: RouteKey
    optimizer_params: dict[str, Any] = field(default_factory=dict)


Op = Union[ForwardOp, ForwardBackwardOp, ZeroGradOp, OptimStepOp]
