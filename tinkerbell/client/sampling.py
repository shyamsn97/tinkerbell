"""SamplingClient: submits sample/load/shutdown ops to the API."""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any

import torch
from tinker.types import TensorData

from tinkerbell.client.common import BaseClient, JobHandle
from tinkerbell.types import (
    ActorStatusRequest,
    ActorStatusResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    SampleRequest,
    ShutdownSamplingActorRequest,
    ShutdownSamplingActorResponse,
)
from tinkerbell.types.responses import LogprobsResponse, SampleResponse
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def make_logprobs_tensor(logprobs: list[list[list]], vocab_size: int) -> torch.Tensor:
    seq_len = len(logprobs)
    result = torch.full((seq_len, vocab_size), float("-inf"), dtype=torch.float32)
    for token_pos_idx, token_position in enumerate(logprobs):
        for entry in token_position:
            if isinstance(entry, list) and len(entry) >= 2:
                lp_value, token_id = entry[0], entry[1]
                if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                    result[token_pos_idx, token_id] = lp_value
    return result


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
        logprobs = LogprobsResponse(
            logprobs=TensorData.from_torch(
                torch.tensor(
                    [[sub[0] for sub in item] for item in raw_logprobs],
                    dtype=torch.float32,
                ).squeeze()
            ),
            token_ids=TensorData.from_torch(
                torch.tensor(
                    [[sub[1] for sub in item] for item in raw_logprobs],
                    dtype=torch.int64,
                ).squeeze()
            ),
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
    raw_logprobs = meta_info.get("output_top_logprobs") or result.get("logprobs")

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


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SamplingClient(BaseClient):
    def __init__(
        self,
        server_url: str,
        base_model: str,
        model_name: str | None = None,
        adapter_name: str | None = None,
        timeout: float = 600.0,
    ):
        super().__init__(server_url=server_url, timeout=timeout, base_model=base_model)
        self.model_name = model_name or clean_model_name(base_model)
        self.adapter_name = adapter_name
        self.lora_path: str | None = None

    # ---- status ----

    def get_status(self) -> JobHandle[ActorStatusResponse]:
        req = ActorStatusRequest(model_name=self.model_name)
        return self._submit(
            "/get_sampling_actor_status",
            req.model_dump(exclude_none=True),
            lambda r: ActorStatusResponse(**r),
        )

    def wait_until_ready(
        self,
        poll_interval: float = 1.0,
        verbose: bool = True,
        timeout: float = 900.0,
    ) -> None:
        start = time.time()
        last_status = None
        last_print = start
        actor_name = f"sampling_actor_{clean_model_name(self.model_name)}"

        if verbose:
            logger.info("=" * 80)
            logger.info("Waiting for sampling actor to be ready...")
            logger.info(f"  Base model: {self.base_model}")
            logger.info(f"  Actor name (model_name): {self.model_name}")
            logger.info(f"  Ray actor name: {actor_name}")
            logger.info(f"  Timeout: {timeout}s")
            logger.info(f"To check actor logs: ray logs {actor_name}")
            logger.info("=" * 80)

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                msg = (
                    f"Sampling actor did not become ready within {timeout}s. "
                    f"Last status: {last_status}"
                )
                logger.error(msg)
                raise TimeoutError(msg)

            status = self.get_status().result()
            if verbose and (
                status.status != last_status or (time.time() - last_print) >= 30
            ):
                logger.info(f"[{elapsed:.1f}s] Sampling actor status: {status.status}")
                last_print = time.time()
            last_status = status.status

            if status.status == "ready":
                if verbose:
                    logger.info(f"Sampling actor is ready! (took {elapsed:.1f}s)")
                return
            if status.status == "not_present":
                raise RuntimeError(
                    f"Sampling actor became 'not_present' after {elapsed:.1f}s."
                )
            time.sleep(poll_interval)

    # ---- sampling ----

    def _sample_payload(self, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("model_name", self.model_name)
        if self.adapter_name and self.lora_path and "lora_path" not in kwargs:
            kwargs["lora_path"] = self.lora_path
        req = SampleRequest(**kwargs)
        return req.model_dump(exclude_none=True)

    def _sample_parser(self):
        tokenizer = self.get_tokenizer()
        return functools.partial(
            parse_sample_response, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer
        )

    def sample(self, **kwargs) -> JobHandle[SampleResponse]:
        return self._submit(
            "/sample", self._sample_payload(**kwargs), self._sample_parser()
        )

    def sample_batch(self, batch_kwargs) -> list[JobHandle[SampleResponse]]:
        return [self.sample(**kw) for kw in batch_kwargs]

    # ---- checkpoint ----

    def load_checkpoint(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> JobHandle[LoadCheckpointResponse]:
        self.lora_path = os.path.normpath(checkpoint_path)
        req = LoadCheckpointRequest(
            model_name=self.model_name,
            checkpoint_path=checkpoint_path,
            pin_lora=pin_lora,
        )
        return self._submit(
            "/load_checkpoint",
            req.model_dump(exclude_none=True),
            lambda r: LoadCheckpointResponse(**r),
        )

    def get_lora_info(self) -> dict[str, Any]:
        req = ActorStatusRequest(model_name=self.model_name)
        return self.transport.submit(
            "/get_lora_info", req.model_dump(exclude_none=True)
        )

    def shutdown(self) -> JobHandle[ShutdownSamplingActorResponse]:
        req = ShutdownSamplingActorRequest(model_name=self.model_name)
        return self._submit(
            "/shutdown_sampling_actor",
            req.model_dump(exclude_none=True),
            lambda r: ShutdownSamplingActorResponse(**r),
        )
