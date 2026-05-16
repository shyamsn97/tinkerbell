"""Ray driver session for the small Tinkerbell API."""

from __future__ import annotations

from typing import Any

import ray

from tinkerbell.runtime.resources import ActorResources
from tinkerbell.sampling.sampler import Sampler, attach_sampler, create_sampler
from tinkerbell.training.group import TrainGroup, attach_train_group, create_train_group


class Session:
    """Owns the Ray connection and creates local trainer/sampler proxies."""

    def __init__(self, namespace: str = "tinkerbell"):
        self.namespace = namespace

    def trainer(
        self,
        *,
        base_model: str,
        world_size: int = 1,
        tp_size: int | None = None,
        model_name: str | None = None,
        adapter_name: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        parallelize_plan: dict[str, str] | None = None,
        lora_config: dict[str, Any] | None = None,
        initialize_base_model: bool = False,
        resources: ActorResources | None = None,
        name: str | None = None,
        detached: bool = False,
    ) -> TrainGroup:
        return create_train_group(
            base_model=base_model,
            world_size=tp_size or world_size,
            model_name=model_name,
            adapter_name=adapter_name,
            model_kwargs=model_kwargs,
            parallelize_plan=parallelize_plan,
            lora_config=lora_config,
            initialize_base_model=initialize_base_model,
            resources=resources,
            namespace=self.namespace,
            name=name,
            detached=detached,
        )

    def sampler(
        self,
        *,
        base_model: str,
        tp_size: int = 1,
        model_name: str | None = None,
        adapter_name: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        resources: ActorResources | None = None,
        sample_concurrency: int = 64,
        name: str | None = None,
        detached: bool = False,
    ) -> Sampler:
        return create_sampler(
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
            tp_size=tp_size,
            engine_kwargs=engine_kwargs,
            resources=resources,
            sample_concurrency=sample_concurrency,
            namespace=self.namespace,
            name=name,
            detached=detached,
        )

    def attach_trainer(
        self,
        *,
        name: str,
        world_size: int,
        base_model: str,
        model_name: str | None = None,
        adapter_name: str | None = None,
    ) -> TrainGroup:
        return attach_train_group(
            name=name,
            world_size=world_size,
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
            namespace=self.namespace,
        )

    def attach_sampler(
        self,
        *,
        name: str,
        base_model: str,
        model_name: str | None = None,
        adapter_name: str | None = None,
    ) -> Sampler:
        return attach_sampler(
            name=name,
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
            namespace=self.namespace,
        )


def init(
    address: str | None = None,
    namespace: str = "tinkerbell",
    ignore_reinit_error: bool = True,
    **ray_init_kwargs,
) -> Session:
    if not ray.is_initialized():
        ray.init(
            address=address,
            namespace=namespace,
            ignore_reinit_error=ignore_reinit_error,
            **ray_init_kwargs,
        )
    return Session(namespace=namespace)


__all__ = ["Session", "init"]
