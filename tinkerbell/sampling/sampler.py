"""Ray-native sampling proxy."""

from __future__ import annotations

import functools
import os
import time
from typing import Any

import torch
from tinker.types import TensorData

from tinkerbell.runtime.futures import Future
from tinkerbell.runtime.resources import ActorResources
from tinkerbell.types.responses import (
    ActorStatusResponse,
    LoadCheckpointResponse,
    LogprobsResponse,
    SampleResponse,
    ShutdownSamplingActorResponse,
)
from tinkerbell.utils import clean_model_name


def _to_jsonable(value):
    if isinstance(value, TensorData):
        return value.to_torch().tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def parse_single_sample(
    output: str,
    finish_reason: str | None,
    raw_logprobs: list | None,
    tokens_generated: int | None,
    meta_info: dict | None,
    tokenizer=None,
) -> SampleResponse:
    logprobs = None
    if raw_logprobs:
        lp_values, token_ids = _coerce_logprobs(raw_logprobs)
        logprobs = LogprobsResponse(
            logprobs=TensorData.from_torch(
                torch.tensor(lp_values, dtype=torch.float32)
            ),
            token_ids=TensorData.from_torch(torch.tensor(token_ids, dtype=torch.int64)),
        )

    output_token_ids = None
    if tokenizer is not None and output:
        output_token_ids = tokenizer.encode(output, add_special_tokens=False)

    return SampleResponse(
        output=output,
        tokens_generated=tokens_generated,
        logprobs=logprobs,
        finish_reason=finish_reason,
        meta_info=meta_info or None,
        output_token_ids=output_token_ids,
    )


def parse_sample_response(
    result: dict[str, Any], vocab_size: int, tokenizer=None
) -> SampleResponse:
    meta_info = result.get("meta_info") or {}
    raw_logprobs = (
        meta_info.get("output_top_logprobs")
        or meta_info.get("output_token_logprobs")
        or meta_info.get("logprobs")
        or result.get("logprobs")
        or result.get("top_logprobs")
    )
    outputs = result.get("outputs", [])
    output = outputs[0] if outputs else ""
    finish_reasons = result.get("finish_reasons")
    finish_reason = finish_reasons[0] if finish_reasons else None
    return parse_single_sample(
        output=output,
        finish_reason=finish_reason,
        raw_logprobs=raw_logprobs,
        tokens_generated=result.get("tokens_generated"),
        meta_info=meta_info,
        tokenizer=tokenizer,
    )


def _coerce_logprobs(raw_logprobs: list) -> tuple[list[float], list[int]]:
    """Normalize SGLang logprob shapes to one value/token id per output token.

    SGLang may return either:
      - output_token_logprobs: [(logprob, token_id), ...]
      - output_top_logprobs: [[(logprob, token_id), ...], ...]
    For GRPO we only need the sampled token's logprob, so top-1 is enough.
    """
    if (
        len(raw_logprobs) == 1
        and isinstance(raw_logprobs[0], list)
        and raw_logprobs[0]
        and isinstance(raw_logprobs[0][0], (list, tuple, dict))
    ):
        raw_logprobs = raw_logprobs[0]

    values: list[float] = []
    token_ids: list[int] = []
    for item in raw_logprobs:
        if isinstance(item, dict):
            logprob = item.get("logprob")
            token_id = item.get("token_id") or item.get("token")
            if logprob is None or token_id is None:
                continue
            values.append(float(logprob))
            token_ids.append(int(token_id))
            continue
        entry = item[0] if item and isinstance(item[0], (list, tuple)) else item
        if isinstance(entry, dict):
            logprob = entry.get("logprob")
            token_id = entry.get("token_id") or entry.get("token")
            if logprob is None or token_id is None:
                continue
            values.append(float(logprob))
            token_ids.append(int(token_id))
            continue
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        logprob, token_id = entry[0], entry[1]
        if logprob is None or token_id is None:
            continue
        values.append(float(logprob))
        token_ids.append(int(token_id))
    return values, token_ids


class Sampler:
    """Local proxy over one SGLang Ray actor."""

    def __init__(
        self,
        *,
        actor: Any,
        base_model: str,
        model_name: str,
        adapter_name: str | None = None,
    ):
        self.actor = actor
        self.base_model = base_model
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.lora_path: str | None = None
        self._tokenizer = None

    def ready(self) -> Future[bool]:
        return Future(self.actor.is_ready.remote())

    def get_status(self) -> Future[ActorStatusResponse]:
        return Future(
            self.actor.is_server_alive.remote(),
            combine=lambda xs: ActorStatusResponse(
                status="ready" if xs[0] else "not_present",
                message=None,
            ),
        )

    def wait_until_ready(
        self,
        poll_interval: float = 1.0,
        timeout: float = 900.0,
        verbose: bool = True,
    ) -> None:
        del verbose
        start = time.time()
        while True:
            if self.get_status().result().status == "ready":
                return
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Sampler {self.model_name} did not become ready within {timeout}s"
                )
            time.sleep(poll_interval)

    def _payload(self, **kwargs) -> dict[str, Any]:
        # GRPO needs sampled-token logprobs. The old pydantic SampleRequest
        # supplied these defaults; keep them in the small client/server path.
        kwargs.setdefault("return_logprob", True)
        kwargs.setdefault("top_logprobs_num", 1)
        if self.adapter_name and self.lora_path and "lora_path" not in kwargs:
            kwargs["lora_path"] = self.lora_path
        kwargs.pop("model_name", None)
        return {k: _to_jsonable(v) for k, v in kwargs.items() if v is not None}

    def _sample_parser(self):
        tokenizer = self.get_tokenizer()
        return functools.partial(
            parse_sample_response, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
        )

    def sample(
        self, text: str | list[str] | None = None, **kwargs
    ) -> Future[SampleResponse]:
        if text is not None:
            kwargs["text"] = text
        payload = self._payload(**kwargs)
        parser = self._sample_parser()
        return Future(
            self.actor.sample.remote(payload), combine=lambda xs: parser(xs[0])
        )

    def sample_batch(
        self, batch_kwargs: list[dict[str, Any]]
    ) -> list[Future[SampleResponse]]:
        return [self.sample(**kw) for kw in batch_kwargs]

    def load_checkpoint(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> Future[LoadCheckpointResponse]:
        self.lora_path = os.path.normpath(checkpoint_path)

        def combine(xs: list[dict[str, Any]]) -> LoadCheckpointResponse:
            result = xs[0] or {}
            return LoadCheckpointResponse(
                model_name=self.model_name,
                success=True,
                message=f"Checkpoint loaded from {checkpoint_path}",
                lora_name=result.get("lora_name"),
            )

        return Future(
            self.actor.update_weights_from_disk.remote(
                checkpoint_path=checkpoint_path,
                load_format=None,
                pin_lora=pin_lora,
            ),
            combine=combine,
        )

    def get_lora_info(self) -> dict[str, Any]:
        import ray

        return ray.get(self.actor.get_lora_info.remote())

    def shutdown(self) -> Future[ShutdownSamplingActorResponse]:
        return Future(
            self.actor.shutdown.remote(),
            combine=lambda _: ShutdownSamplingActorResponse(
                model_name=self.model_name,
                success=True,
                message=f"Sampler {self.model_name} shut down",
            ),
        )

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or 0
        return self._tokenizer


def create_sampler(
    *,
    base_model: str,
    model_name: str | None = None,
    adapter_name: str | None = None,
    tp_size: int = 1,
    engine_kwargs: dict[str, Any] | None = None,
    resources: ActorResources | None = None,
    sample_concurrency: int = 64,
    namespace: str = "tinkerbell",
    name: str | None = None,
    detached: bool = False,
) -> Sampler:
    from tinkerbell.sampling.actor import SGLangSamplingActor

    model_name = model_name or clean_model_name(base_model)
    resources = resources or ActorResources(num_gpus=tp_size)
    opts = resources.actor_options()
    opts.update(
        {
            "max_concurrency": sample_concurrency + 2,
            "namespace": namespace,
        }
    )
    if name is not None:
        opts["name"] = name
        if detached:
            opts["lifetime"] = "detached"

    actor = SGLangSamplingActor.options(**opts).remote(
        base_model=base_model,
        tp_size=tp_size,
        engine_kwargs=engine_kwargs or {},
    )
    return Sampler(
        actor=actor,
        base_model=base_model,
        model_name=model_name,
        adapter_name=adapter_name,
    )


def attach_sampler(
    *,
    name: str,
    base_model: str,
    model_name: str | None = None,
    adapter_name: str | None = None,
    namespace: str = "tinkerbell",
) -> Sampler:
    import ray

    return Sampler(
        actor=ray.get_actor(name, namespace=namespace),
        base_model=base_model,
        model_name=model_name or clean_model_name(base_model),
        adapter_name=adapter_name,
    )


__all__ = [
    "Sampler",
    "attach_sampler",
    "create_sampler",
    "parse_sample_response",
]
