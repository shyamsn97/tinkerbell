import logging
import time
from typing import Any

from tinkerbell.client.base import AsyncTinkerbellFuture, BaseClient, TinkerbellFuture
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

logger = logging.getLogger(__name__)


class SamplingClient(BaseClient):
    """Client for interacting with the Tinkerbell sampling service."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        adapter_name: str | None = None,
        timeout: float = 600.0,
    ):
        super().__init__(server_url, timeout)
        self.model_id = model_id
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
        actor_name = self.model_id.replace("/", "_").replace(":", "_").lower()
        actor_name = f"sampling_actor_{actor_name}"

        if verbose:
            logger.info("=" * 80)
            logger.info("Waiting for sampling actor to be ready...")
            logger.info(f"  Model: {self.model_id}")
            logger.info(f"  Actor name: {actor_name}")
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
        request = ActorStatusRequest(model_id=self.model_id)
        response = self.client.post(
            "/get_sampling_actor_status",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        remote_future_dict = response.json()

        # Parse result
        def _parse_result(result: dict[str, Any]) -> ActorStatusResponse:
            return ActorStatusResponse(**result)

        return self.create_future_from_request_id(
            remote_future=RemoteFuture(**remote_future_dict),
            parse_result_fn=_parse_result,
        )

    def sample(self, *args, **kwargs) -> TinkerbellFuture[SampleResponse]:
        """Sample text. Uses lora_path only if adapter_name is set (LoRA mode). takes in args and kwargs like SampleRequest."""
        if "model_id" not in kwargs:
            kwargs["model_id"] = self.model_id
        # Only pass lora_path for LoRA adapters (when adapter_name is set)
        if self.adapter_name and self.lora_path and "lora_path" not in kwargs:
            kwargs["lora_path"] = self.lora_path

        request = SampleRequest(*args, **kwargs)
        response = self.client.post(
            "/sample", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()

        return self.create_future_from_request_id(
            remote_future=RemoteFuture(**response.json()),
            parse_result_fn=lambda result: SampleResponse(**result),
        )

    async def sample_async(
        self, *args, **kwargs
    ) -> AsyncTinkerbellFuture[SampleResponse]:
        """Async: Sample text. Uses lora_path only if adapter_name is set."""
        if "model_id" not in kwargs:
            kwargs["model_id"] = self.model_id
        if self.adapter_name and self.lora_path and "lora_path" not in kwargs:
            kwargs["lora_path"] = self.lora_path

        request = SampleRequest(*args, **kwargs)
        response = await self.async_client.post(
            "/sample", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()

        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(**response.json()),
            server_url=self.server_url,
            result_parser=lambda result: SampleResponse(**result),
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def load_checkpoint(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> TinkerbellFuture[LoadCheckpointResponse]:
        """Load checkpoint from disk. Stores lora_name for subsequent samples."""
        import os

        request = LoadCheckpointRequest(
            model_id=self.model_id, checkpoint_path=checkpoint_path, pin_lora=pin_lora
        )
        response = self.client.post(
            "/load_checkpoint", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()

        # Store lora_name (basename) for sample requests - SGLang uses basename as adapter name
        self.lora_path = os.path.basename(os.path.normpath(checkpoint_path))

        return self.create_future_from_request_id(
            remote_future=RemoteFuture(**response.json()),
            parse_result_fn=lambda result: LoadCheckpointResponse(**result),
        )

    async def load_checkpoint_async(
        self, checkpoint_path: str, pin_lora: bool = False
    ) -> AsyncTinkerbellFuture[LoadCheckpointResponse]:
        """Async: Load checkpoint. First await submits, second await gets result."""
        import os

        request = LoadCheckpointRequest(
            model_id=self.model_id, checkpoint_path=checkpoint_path, pin_lora=pin_lora
        )
        response = await self.async_client.post(
            "/load_checkpoint", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()

        # Store lora_name (basename) for sample requests
        self.lora_path = os.path.basename(os.path.normpath(checkpoint_path))

        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(**response.json()),
            server_url=self.server_url,
            result_parser=lambda result: LoadCheckpointResponse(**result),
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def shutdown(self) -> TinkerbellFuture[ShutdownSamplingActorResponse]:
        """
                Shutdown the sampling actor.

                Returns:
        TinkerbellFuture[ShutdownSamplingActorResponse] - call .result() to poll for the response
        """
        # Send request immediately
        request = ShutdownSamplingActorRequest(model_id=self.model_id)

        response = self.client.post(
            "/shutdown_sampling_actor",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        remote_future_dict = response.json()

        # Parse result
        def _parse_result(result: dict[str, Any]) -> ShutdownSamplingActorResponse:
            return ShutdownSamplingActorResponse(**result)

        return self.create_future_from_request_id(
            remote_future=RemoteFuture(**remote_future_dict),
            parse_result_fn=_parse_result,
        )

    def close(self):
        """Close the HTTP client."""
        self.client.close()
        if self._async_client is not None:
            import warnings

            warnings.warn(
                "Async client not closed. Use 'async with' or call await client.aclose()",
                ResourceWarning,
            )

    async def aclose(self):
        """Close both sync and async HTTP clients."""
        self.client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
