"""Optimizer factory + adapter-change-aware OptimizerManager.

Extracted from the old `TrainingActor._get_optimizer` /
`_get_or_create_optimizer`. Pure Python, no Ray, so it's easy to unit-test
and reuse outside a Ray actor.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch

logger = logging.getLogger(__name__)


OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
}


def build_optimizer(
    params: Iterable[torch.nn.Parameter], config: dict[str, Any]
) -> torch.optim.Optimizer:
    """Construct an optimizer from a {"name": "adamw", "lr": ..., ...} dict.

    Mutates a copy of `config` (pops "name"; forces foreach=False for LoRA safety).
    """
    config = config.copy()
    name = config.pop("name", "adam").lower()
    config.pop("grad_clip_norm", None)
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. Available: {sorted(OPTIMIZER_REGISTRY)}"
        )
    # foreach=True can mis-handle LoRA's partial param set; force False.
    config["foreach"] = False
    return OPTIMIZER_REGISTRY[name](list(params), **config)


class OptimizerManager:
    """Caches a torch.optim.Optimizer and rebuilds it on adapter switches.

    The old code scattered this logic inside TrainingActor; extracting it lets
    the Trainer hold one OptimizerManager per actor. Pure Python.
    """

    def __init__(self):
        self.optimizer: torch.optim.Optimizer | None = None
        self.param_ids: set[int] = set()

    def get(
        self,
        params_fn,  # Callable[[], Iterable[torch.nn.Parameter]]
        config: dict[str, Any],
    ) -> torch.optim.Optimizer:
        current_params = list(params_fn())
        current_ids = {id(p) for p in current_params}

        if self.optimizer is None:
            self.optimizer = build_optimizer(current_params, config)
            self.param_ids = current_ids
            return self.optimizer

        if current_ids != self.param_ids:
            logger.info("Trainable parameters changed, rebuilding optimizer")
            del self.optimizer
            self.optimizer = build_optimizer(current_params, config)
            self.param_ids = current_ids
        elif "lr" in config:
            for group in self.optimizer.param_groups:
                group["lr"] = config["lr"]

        return self.optimizer
