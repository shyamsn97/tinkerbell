"""Ray-native training group proxy."""

from __future__ import annotations

from typing import Any

import ray
from ray.util.placement_group import placement_group

from tinkerbell.renderer import MASK_TOKEN_ID, Renderer, RenderMode, TrainOnWhat
from tinkerbell.runtime.futures import Future
from tinkerbell.runtime.resources import ActorResources
from tinkerbell.training.actor import TrainingActor
from tinkerbell.types.responses import (
    ForwardBackwardResponse,
    ForwardResponse,
    OptimStepResponse,
    PushToHubResponse,
    SaveCheckpointResponse,
    ZeroGradResponse,
)
from tinkerbell.utils import clean_model_name, get_free_port


def _first_rank(values: list[Any]) -> Any:
    return next((v for v in values if v is not None), None)


class TrainGroup:
    """Local proxy over one Ray training actor per TP rank."""

    def __init__(
        self,
        *,
        workers: list[Any],
        base_model: str,
        model_name: str,
        adapter_name: str | None,
        namespace: str,
        placement_group_ref: Any | None = None,
    ):
        self.workers = workers
        self.base_model = base_model
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.namespace = namespace
        self.placement_group = placement_group_ref
        self._tokenizer = None
        self._renderer: Renderer | None = None

    def ready(self) -> Future[bool]:
        return Future(
            [w.setup.remote() for w in self.workers], combine=lambda xs: all(xs)
        )

    def wait_until_ready(self) -> None:
        self.ready().result()

    def forward(
        self,
        data: list[Any],
        forward_kwargs: dict[str, Any] | None = None,
    ) -> Future[ForwardResponse]:
        refs = [
            w.forward.remote(data=data, forward_kwargs=forward_kwargs or {})
            for w in self.workers
        ]
        return Future(
            refs,
            combine=lambda xs: ForwardResponse(**(_first_rank(xs) or {})),
        )

    def forward_backward(
        self,
        data: list[Any],
        forward_kwargs: dict[str, Any] | None = None,
        return_logprobs: bool = False,
        zero_grad: bool = False,
        optimizer_params: dict[str, Any] | None = None,
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, float] | None = None,
    ) -> Future[ForwardBackwardResponse]:
        refs = []
        if zero_grad:
            refs.extend(w.zero_grad.remote() for w in self.workers)

        fb_refs = [
            w.forward_backward.remote(
                data=data,
                adapter_name=self.adapter_name,
                forward_kwargs=forward_kwargs or {},
                loss_fns=[loss_fn] * len(data),
                loss_fn_config=loss_fn_config,
            )
            for w in self.workers
        ]
        refs.extend(fb_refs)

        if optimizer_params is not None:
            refs.extend(
                w.optim_step.remote(
                    adapter_name=self.adapter_name,
                    optimizer_params=optimizer_params,
                )
                for w in self.workers
            )

        def combine(values: list[Any]) -> ForwardBackwardResponse:
            offset = len(self.workers) if zero_grad else 0
            fb_values = values[offset : offset + len(self.workers)]
            result = _first_rank(fb_values) or {}
            return ForwardBackwardResponse(model_name=self.model_name, **result)

        return Future(refs, combine=combine)

    def zero_grad(self) -> Future[ZeroGradResponse]:
        refs = [w.zero_grad.remote() for w in self.workers]
        return Future(
            refs,
            combine=lambda _: ZeroGradResponse(
                model_name=self.model_name,
                message="gradients zeroed",
            ),
        )

    def optim_step(
        self, optimizer_params: dict[str, Any] | None = None
    ) -> Future[OptimStepResponse]:
        refs = [
            w.optim_step.remote(
                adapter_name=self.adapter_name,
                optimizer_params=optimizer_params or {},
            )
            for w in self.workers
        ]
        return Future(
            refs,
            combine=lambda _: OptimStepResponse(
                model_name=self.model_name,
                message="optimizer stepped",
            ),
        )

    def save_checkpoint(self, checkpoint_path: str) -> Future[SaveCheckpointResponse]:
        refs = [
            w.save_checkpoint.remote(checkpoint_path, self.adapter_name)
            for w in self.workers
        ]
        return Future(
            refs,
            combine=lambda _: SaveCheckpointResponse(
                model_name=self.model_name,
                success=True,
                path=checkpoint_path,
            ),
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
        push_kwargs: dict[str, Any] | None = None,
    ) -> Future[PushToHubResponse]:
        refs = [
            w.push_to_hub.remote(
                repo_id=repo_id,
                adapter_name=self.adapter_name,
                token=token,
                private=private,
                commit_message=commit_message,
                push_kwargs=push_kwargs or {},
            )
            for w in self.workers
        ]
        return Future(
            refs,
            combine=lambda _: PushToHubResponse(
                model_name=self.model_name,
                success=True,
                repo_id=repo_id,
            ),
        )

    def shutdown(self) -> Future[None]:
        refs = [w.cleanup.remote() for w in self.workers]
        return Future(refs, combine=lambda _: None)

    def save_weights_and_get_sampling_client(
        self,
        checkpoint_path: str,
        tp_size: int | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        wait_until_ready: bool = False,
    ):
        from tinkerbell.sampling.sampler import create_sampler

        self.save_checkpoint(checkpoint_path).result()
        is_lora = self.adapter_name is not None
        final_kwargs = dict(engine_kwargs or {})
        actor_base_model = self.base_model
        if is_lora:
            final_kwargs["lora_paths"] = [checkpoint_path]
        else:
            actor_base_model = checkpoint_path
            final_kwargs["enable_lora"] = False

        sampler = create_sampler(
            base_model=actor_base_model,
            model_name=self.model_name,
            adapter_name=self.adapter_name,
            tp_size=tp_size or 1,
            engine_kwargs=final_kwargs,
            namespace=self.namespace,
        )
        if wait_until_ready:
            sampler.wait_until_ready()
        if is_lora:
            sampler.load_checkpoint(checkpoint_path).result()
        return sampler

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or 0
        return self._tokenizer

    def get_renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer(self.get_tokenizer())
        return self._renderer

    def render(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        mode: RenderMode = RenderMode.TRAINING,
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        continue_final_message: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        return self.get_renderer().render(
            messages=messages,
            mode=mode,
            train_on_what=train_on_what,
            mask_value=mask_value,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def build_chat_samples(
        self,
        messages: list[list[dict[str, str]]] | list[dict[str, str]],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value: int = MASK_TOKEN_ID,
        include_labels: bool = True,
        **kwargs,
    ):
        mode = RenderMode.TRAINING if include_labels else RenderMode.INFERENCE
        return self.render(messages, mode, train_on_what, mask_value, **kwargs)


def create_train_group(
    *,
    base_model: str,
    world_size: int = 1,
    model_name: str | None = None,
    adapter_name: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    parallelize_plan: dict[str, str] | None = None,
    lora_config: dict[str, Any] | None = None,
    initialize_base_model: bool = False,
    resources: ActorResources | None = None,
    namespace: str = "tinkerbell",
    name: str | None = None,
    detached: bool = False,
) -> TrainGroup:
    model_name = model_name or clean_model_name(base_model)
    resources = resources or ActorResources()
    master_addr = "127.0.0.1"
    master_port = str(get_free_port())

    bundles = [
        {
            "CPU": resources.num_cpus,
            "GPU": resources.num_gpus,
            **resources.resources,
        }
        for _ in range(world_size)
    ]
    pg = placement_group(bundles, strategy="STRICT_PACK")
    ray.get(pg.ready())

    workers = []
    for rank in range(world_size):
        opts = resources.actor_options()
        opts.update(
            {
                "placement_group": pg,
                "placement_group_bundle_index": rank,
                "namespace": namespace,
            }
        )
        if name is not None:
            opts["name"] = f"{name}_{rank}"
            opts["lifetime"] = "detached" if detached else None
        opts = {k: v for k, v in opts.items() if v is not None}
        workers.append(
            TrainingActor.options(**opts).remote(
                rank=rank,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                base_model=base_model,
                model_name=model_name,
                model_kwargs=model_kwargs or {},
                parallelize_plan=parallelize_plan or {},
                lora_config=lora_config,
                adapter_name=adapter_name,
                initialize_base_model=initialize_base_model,
            )
        )

    return TrainGroup(
        workers=workers,
        base_model=base_model,
        model_name=model_name,
        adapter_name=adapter_name,
        namespace=namespace,
        placement_group_ref=pg,
    )


def attach_train_group(
    *,
    name: str,
    world_size: int,
    base_model: str,
    model_name: str | None = None,
    adapter_name: str | None = None,
    namespace: str = "tinkerbell",
) -> TrainGroup:
    workers = [
        ray.get_actor(f"{name}_{rank}", namespace=namespace)
        for rank in range(world_size)
    ]
    return TrainGroup(
        workers=workers,
        base_model=base_model,
        model_name=model_name or clean_model_name(base_model),
        adapter_name=adapter_name,
        namespace=namespace,
    )


__all__ = ["TrainGroup", "create_train_group", "attach_train_group"]
