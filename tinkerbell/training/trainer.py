"""Trainer: pure-Python core of a single training rank.

Wraps an `LLM` + an `OptimizerManager` and exposes imperative methods
(`forward`, `forward_backward`, `zero_grad`, `optim_step`) that operate on
ready-to-go tensors. No Ray, no distributed init here — the Ray-side
`TrainingActor` handles those, then delegates to this class.

The separation lets us test the gradient math + loss wiring without
standing up a Ray cluster, and lets `TrainingActor` stay tiny.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import torch
from tinker.types import Datum, LoraConfig, LossFnType

from tinkerbell.training.llm import LLM
from tinkerbell.training.loss import LOSSES
from tinkerbell.training.optimizer import OptimizerManager

logger = logging.getLogger(__name__)


def coerce_lora_config(config: LoraConfig | dict[str, Any] | None) -> LoraConfig | None:
    if config is None:
        return None
    return LoraConfig(**config) if isinstance(config, dict) else config


class Trainer:
    """Single-rank imperative trainer.

    Lifecycle:
        t = Trainer(...)
        t.setup_model()            # loads weights, attaches LoRA
        t.forward_backward(...)    # many calls; accumulates grads
        t.optim_step(...)          # applies grads
        t.save_model(path)         # snapshots to disk
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        base_model: str,
        model_kwargs: dict[str, Any] | None = None,
        parallelize_plan: dict[str, str] | None = None,
        lora_config: LoraConfig | dict[str, Any] | None = None,
        adapter_name: str | None = None,
        initialize_base_model: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.base_model = base_model
        self.model_kwargs = model_kwargs or {}
        self.parallelize_plan = parallelize_plan or {}
        self.lora_config = coerce_lora_config(lora_config)
        self.adapter_name = adapter_name
        self.initialize_base_model = initialize_base_model

        self.llm: LLM | None = None
        self.optim_manager = OptimizerManager()
        self.ready = False

    def setup_model(self) -> None:
        if self.ready:
            return
        self.llm = LLM(
            rank=self.rank,
            world_size=self.world_size,
            base_model=self.base_model,
            model_kwargs=self.model_kwargs,
            parallelize_plan=self.parallelize_plan,
            lora_config=self.lora_config,
            adapter_name=self.adapter_name,
            initialize_base_model=self.initialize_base_model,
        )
        self.ready = True

    def trainable_params(self):
        return (p for p in self.llm.model.parameters() if p.requires_grad)

    def clear_grads(self) -> None:
        for p in self.trainable_params():
            if p.grad is not None:
                p.grad = None

    def add_adapter(self, adapter_name: str, lora_config: dict[str, Any]) -> None:
        self.llm.add_adapter(adapter_name, coerce_lora_config(lora_config))

    def set_active_adapter(self, adapter_name: str) -> None:
        self.llm.set_active_adapter(adapter_name)

    def adapters(self) -> list[str]:
        return list(self.llm.adapters.keys())

    def forward(
        self,
        data: list[Datum],
        forward_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        prepared = self.llm.prepare_inputs(data, device)
        output = self.llm.forward(
            model_inputs=prepared["model_input"],
            with_grad=True,
            forward_kwargs=forward_kwargs or {},
        )
        return output["logprobs"]

    def forward_backward(
        self,
        data: list[Datum],
        adapter_name: str | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        loss_fns: list[LossFnType] | None = None,
        loss_fn_config: dict[str, float] | None = None,
    ) -> dict[str, Any] | None:
        """Run one forward+backward pass. Returns per-example losses + grad sums on rank 0."""
        loss_fns = loss_fns or ["cross_entropy"]
        if len(loss_fns) != len(data):
            raise ValueError(
                f"loss_fns length ({len(loss_fns)}) must match data ({len(data)})"
            )
        for fn in loss_fns:
            if fn not in LOSSES:
                raise ValueError(f"Unknown loss function: {fn}")

        if adapter_name and self.llm.adapters:
            self.llm.set_active_adapter(adapter_name)

        self.llm.train()
        device = torch.cuda.current_device()
        prepared = self.llm.prepare_inputs(data, device)

        try:
            output = self.llm.forward(
                model_inputs=prepared["model_input"],
                with_grad=True,
                forward_kwargs=forward_kwargs or {},
            )
            logprobs = output["logprobs"]

            extras = loss_fn_config or {}
            if len(set(loss_fns)) == 1:
                per_batch_losses = LOSSES[loss_fns[0]](
                    logprobs=logprobs,
                    **prepared["loss_fn_inputs"],
                    **extras,
                )
            else:
                per_batch_losses = torch.stack(
                    [
                        LOSSES[fn](
                            logprobs=logprobs[i : i + 1],
                            **{
                                k: v[i : i + 1]
                                for k, v in prepared["loss_fn_inputs"].items()
                            },
                            **extras,
                        )
                        for i, fn in enumerate(loss_fns)
                    ]
                ).squeeze()

            loss_mean = per_batch_losses.mean()
            loss_mean.backward()

            loss_values = (
                [loss.item() for loss in per_batch_losses] if self.rank == 0 else None
            )

            del output, prepared, per_batch_losses, loss_mean

            if self.rank != 0:
                return None

            sum_gradient: dict[str, float] = {}
            with torch.no_grad():
                for name, param in self.llm.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        sum_gradient[name] = param.grad.detach().sum().item()

            return {"loss": loss_values, "sum_gradient": sum_gradient}
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            logger.error(
                f"CUDA OOM in forward_backward (rank {self.rank}): {e}", exc_info=True
            )
            raise RuntimeError(f"CUDA OOM in forward_backward: {e}") from e

    def zero_grad(self) -> None:
        if self.llm is None or self.llm.model is None:
            logger.warning("Model not initialized yet, skipping zero_grad")
            return
        self.clear_grads()

    def optim_step(
        self,
        adapter_name: str | None = None,
        optimizer_params: dict[str, Any] | None = None,
    ) -> None:
        if adapter_name and self.llm.adapters:
            self.llm.set_active_adapter(adapter_name)
        optimizer = self.optim_manager.get(
            self.trainable_params, optimizer_params or {}
        )
        optimizer.step()
        self.clear_grads()

    def save_model(self, path: str, adapter_name: str | None = None) -> None:
        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
        self.llm.save_model(path, adapter_name=adapter_name)

    def push_to_hub(
        self,
        repo_id: str,
        adapter_name: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
        push_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """BLOCKING HF upload. Must be called from a worker thread on rank 0."""
        if self.rank != 0:
            return
        if adapter_name and self.llm.adapters:
            self.llm.set_active_adapter(adapter_name)
        if not hasattr(self.llm.model, "push_to_hub"):
            raise RuntimeError("Model does not support push_to_hub")
        push_params = {
            "repo_id": repo_id,
            "token": token,
            "private": private,
            "commit_message": commit_message,
            **(push_kwargs or {}),
        }
        if adapter_name and self.llm.adapters:
            push_params["adapter_name"] = adapter_name

        logger.info(f"Pushing model to HF Hub repo={repo_id}...")
        self.llm.model.push_to_hub(**push_params)
        if self.llm.tokenizer is not None:
            try:
                self.llm.tokenizer.push_to_hub(
                    repo_id=repo_id,
                    token=token,
                    private=private,
                    commit_message=commit_message,
                )
            except Exception as e:
                logger.warning(f"Failed to push tokenizer: {e}")
        logger.info(f"Finished pushing model to HF Hub repo={repo_id}")

    def cleanup_workdir(self, path: str) -> None:
        """Remove a checkpoint directory on rank 0."""
        if self.rank == 0 and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
