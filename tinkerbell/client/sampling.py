import functools
import logging
import time
from typing import Any

import torch

from tinkerbell.client.base import AsyncTinkerbellFuture, BaseClient, TinkerbellFuture

# from tinkerbell.types.data import TensorData
from tinkerbell.types.requests import (
    ActorStatusRequest,
    LoadCheckpointRequest,
    SampleRequest,
    ShutdownSamplingActorRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    LoadCheckpointResponse,
    RemoteFuture,
    SampleResponse,
    ShutdownSamplingActorResponse,
)
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


def make_logprobs_tensor(logprobs: list[list[list]], vocab_size: int) -> torch.Tensor:
    """
    Convert logprobs from format [[logprob, token_id, None], ...] to a tensor.

    Args:
        logprobs: List of lists, where each inner list contains top-k logprobs
                in format [logprob_value, token_id, None]
        vocab_size: Size of the vocabulary

    Returns:
        Tensor of shape (seq_len, vocab_size) containing logprob values
        at the correct token_id positions, with -inf elsewhere
    """
    seq_len = len(logprobs)
    # Initialize with -inf (log(0) = no probability mass).
    # NOTE: For importance sampling, ensure you request logprobs for the actual
    # sampled tokens via token_ids_logprob, otherwise exp(x - (-inf)) = inf!
    result = torch.full((seq_len, vocab_size), float("-inf"), dtype=torch.float32)

    # Fill in the logprobs at the correct token_id positions
    for token_pos_idx, token_position in enumerate(logprobs):
        for entry in token_position:
            if isinstance(entry, list) and len(entry) >= 2:
                # Extract logprob value (first element) and token_id (second element)
                logprob_value = entry[0]
                token_id = entry[1]
                # Place logprob at the correct position
                if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                    result[token_pos_idx, token_id] = logprob_value
    return result


def parse_sample_response(result: dict[str, Any], vocab_size: int) -> SampleResponse:
    """Parse the sample response."""
    meta_info = result.get("meta_info") or {}

    # Extract output_token_ids if available - wrap in list for batch format
    output_token_ids = None
    output_token_logprobs = meta_info.get("output_token_logprobs")

    if output_token_logprobs:
        # Single sequence: wrap in a list to match list[list[int]] type
        logprobs = [item[0] for item in output_token_logprobs]
        output_token_ids = [item[1] for item in output_token_logprobs]

    # Extract logprobs - try output_top_logprobs first, fall back to top-level logprobs
    raw_logprobs = meta_info.get("output_top_logprobs") or result.get("logprobs")
    # if raw_logprobs is not None:
    #     logprobs = TensorData.from_torch(make_logprobs_tensor(raw_logprobs, vocab_size))

    return SampleResponse(
        outputs=result.get("outputs", []),
        tokens_generated=result.get("tokens_generated"),
        logprobs=logprobs,
        top_logprobs=raw_logprobs,
        output_token_ids=output_token_ids,
        finish_reasons=result.get("finish_reasons"),
        meta_info=meta_info or None,
    )


class SamplingClient(BaseClient):
    """Client for interacting with the Tinkerbell sampling service."""

    def __init__(
        self,
        server_url: str,
        base_model: str,
        model_name: str | None = None,
        adapter_name: str | None = None,
        timeout: float = 600.0,
    ):
        """Initialize sampling client.

        Args:
            server_url: URL of the tinkerbell server
            base_model: HuggingFace model path (e.g., "Qwen/Qwen3-0.6B") - for tokenizer
            model_name: Actor group name for routing. Defaults to cleaned base_model.
            adapter_name: LoRA adapter name (optional)
            timeout: Request timeout in seconds
        """
        super().__init__(server_url=server_url, timeout=timeout, base_model=base_model)
        self.model_name = (
            model_name if model_name else clean_model_name(base_model)
        )  # Actor name for routing
        self.adapter_name = adapter_name
        self.lora_path: str | None = None  # Set after load_checkpoint

    def wait_until_ready(
        self,
        poll_interval: float = 1.0,
        verbose: bool = True,
        timeout: float = 900.0,
    ) -> None:
        """
        Poll the server until the sampling actor is ready.

        Args:
            poll_interval: Time between status checks in seconds
            verbose: If True, prints status updates
            timeout: Maximum time to wait in seconds (default: 15 minutes)
        """
        start_time = time.time()
        last_status = None
        last_print_time = start_time

        # Compute actor name for debugging
        actor_name = self.model_name.replace("/", "_").replace(":", "_").lower()
        actor_name = f"sampling_actor_{actor_name}"

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
            elapsed = time.time() - start_time

            if elapsed > timeout:
                error_msg = (
                    f"Sampling actor did not become ready within {timeout}s. "
                    f"Last status: {last_status}\n"
                    f"Actor name: {actor_name}\n"
                    f"Check Ray logs with: ray logs {actor_name}\n"
                    f"Or check all Ray logs at: /tmp/ray/session_latest/logs/"
                )
                logger.error("=" * 80)
                logger.error("TIMEOUT ERROR:")
                logger.error(error_msg)
                logger.error("=" * 80)
                raise TimeoutError(error_msg)

            status = self.get_status().result()

            # Print if status changed or every 30 seconds
            status_changed = status.status != last_status
            should_print_interval = (time.time() - last_print_time) >= 30

            if verbose and (status_changed or should_print_interval):
                logger.info(f"[{elapsed:.1f}s] Sampling actor status: {status.status}")
                last_print_time = time.time()

            last_status = status.status

            if status.status == "ready":
                if verbose:
                    logger.info(f"✓ Sampling actor is ready! (took {elapsed:.1f}s)")
                break
            elif status.status == "not_present":
                error_msg = (
                    f"Sampling actor became 'not_present' after {elapsed:.1f}s. "
                    f"This usually means the actor crashed during initialization or checkpoint loading.\n"
                    f"Actor name: {actor_name}\n"
                    f"Check Ray logs with: ray logs {actor_name}\n"
                    f"Or check all Ray logs at: /tmp/ray/session_latest/logs/"
                )
                logger.error("=" * 80)
                logger.error("ACTOR CRASHED:")
                logger.error(error_msg)
                logger.error("=" * 80)
                raise RuntimeError(error_msg)

            time.sleep(poll_interval)

    def get_status(self) -> TinkerbellFuture[ActorStatusResponse]:
        """
        Get the status of the sampling actor.

        Returns:
            TinkerbellFuture[ActorStatusResponse] - call .result() to poll for the status
        """
        # Send request immediately
        request = ActorStatusRequest(model_name=self.model_name)
        response = self.client.post(
            "/get_sampling_actor_status",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        remote_future_dict = response.json()

        # Parse result
        def _parse_result(result: dict[str, Any]) -> ActorStatusResponse:
            return ActorStatusResponse(**result)

        return self.create_future_from_remote(
            remote_future=RemoteFuture(**remote_future_dict),
            parse_result_fn=_parse_result,
        )

    def _sample_request(self, *args, **kwargs) -> SampleRequest:
        if "model_name" not in kwargs:
            kwargs["model_name"] = self.model_name
        if self.adapter_name and self.lora_path and "lora_path" not in kwargs:
            kwargs["lora_path"] = self.lora_path
        return SampleRequest(*args, **kwargs)

    def _sample_parser(self):
        return functools.partial(
            parse_sample_response, vocab_size=self.get_tokenizer().vocab_size
        )

    def sample(self, *args, **kwargs) -> TinkerbellFuture[SampleResponse]:
        """Sample text. Uses lora_path only if adapter_name is set (LoRA mode)."""
        request = self._sample_request(*args, **kwargs)
        response = self.client.post(
            "/sample", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return self.create_future_from_remote(
            remote_future=RemoteFuture(**response.json()),
            parse_result_fn=self._sample_parser(),
        )

    async def sample_async(
        self, *args, **kwargs
    ) -> AsyncTinkerbellFuture[SampleResponse]:
        """Async: Sample text. Uses lora_path only if adapter_name is set."""
        return await self.create_async_future(
            request=self._sample_request(*args, **kwargs),
            endpoint="/sample",
            parse_result_fn=self._sample_parser(),
        )

    def _load_checkpoint_request(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> LoadCheckpointRequest:
        import os

        # Store lora_name (basename) for sample requests - SGLang uses basename as adapter name
        self.lora_path = os.path.basename(os.path.normpath(checkpoint_path))
        return LoadCheckpointRequest(
            model_name=self.model_name,
            checkpoint_path=checkpoint_path,
            pin_lora=pin_lora,
        )

    def load_checkpoint(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> TinkerbellFuture[LoadCheckpointResponse]:
        """Load checkpoint from disk. Stores lora_name for subsequent samples."""
        request = self._load_checkpoint_request(checkpoint_path, pin_lora)
        response = self.client.post(
            "/load_checkpoint", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return self.create_future_from_remote(
            remote_future=RemoteFuture(**response.json()),
            parse_result_fn=lambda result: LoadCheckpointResponse(**result),
        )

    async def load_checkpoint_async(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> AsyncTinkerbellFuture[LoadCheckpointResponse]:
        """Async: Load checkpoint. First await submits, second await gets result."""
        return await self.create_async_future(
            request=self._load_checkpoint_request(checkpoint_path, pin_lora),
            endpoint="/load_checkpoint",
            parse_result_fn=lambda result: LoadCheckpointResponse(**result),
        )

    def shutdown(self) -> TinkerbellFuture[ShutdownSamplingActorResponse]:
        """
        Shutdown the sampling actor.

        Returns:
            TinkerbellFuture[ShutdownSamplingActorResponse] - call .result() to poll for the response
        """
        request = ShutdownSamplingActorRequest(model_name=self.model_name)

        response = self.client.post(
            "/shutdown_sampling_actor",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        remote_future_dict = response.json()

        # Parse result
        def _parse_result(result: dict[str, Any]) -> ShutdownSamplingActorResponse:
            return ShutdownSamplingActorResponse(**result)

        return self.create_future_from_remote(
            remote_future=RemoteFuture(**remote_future_dict),
            parse_result_fn=_parse_result,
        )
