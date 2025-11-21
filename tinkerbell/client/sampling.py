import logging
import time
from typing import Any

import httpx

from tinkerbell.types.requests import (
    ActorStatusRequest,
    LoadCheckpointRequest,
    SampleRequest,
    ShutdownSamplingActorRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    LoadCheckpointResponse,
    SampleResponse,
    ShutdownSamplingActorResponse,
)
from tinkerbell.client.base import BaseClient, TinkerbellFuture
logger = logging.getLogger(__name__)


class SamplingClient(BaseClient):
    """Client for interacting with the Tinkerbell sampling service."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        timeout: float = 600.0,
    ):
        """
        Initialize the sampling client.

        Args:
            server_url: Server URL of the sampling service
            model_id: ID of the model
            timeout: Request timeout in seconds
        """
        super().__init__(server_url, timeout)
        self.model_id = model_id

    def wait_until_ready(
        self,
        poll_interval: float = 2.0,
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
        remote_future = response.json()
        
        # Parse result
        def _parse_result(result: dict[str, Any]) -> ActorStatusResponse:
            return ActorStatusResponse(**result)
        
        return TinkerbellFuture(
            request_id=remote_future["request_id"],
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=_parse_result,
            poll_interval=0.1,
            timeout=self.timeout,
        )

    def sample(
        self,
        *args,
        **kwargs,
    ) -> TinkerbellFuture[SampleResponse]:
        """
        Sample text from the request.

        Args:
            *args: Positional arguments to pass to the SampleRequest
            **kwargs: Keyword arguments to pass to the SampleRequest

        Returns:
            TinkerbellFuture[SampleResponse] - call .result() to poll for the response
        """
        # Send request immediately
        # Always include model_id from the client
        if "model_id" not in kwargs:
            kwargs["model_id"] = self.model_id

        request = SampleRequest(
            *args,
            **kwargs,
        )

        response = self.client.post(
            "/sample",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        remote_future = response.json()

        # Parse result
        def _parse_result(result: dict[str, Any]) -> SampleResponse:
            return SampleResponse(**result)

        return TinkerbellFuture(
            request_id=remote_future["request_id"],
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=_parse_result,
            poll_interval=0.1,
            timeout=self.timeout,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        pin_lora: bool = False,
    ) -> TinkerbellFuture[LoadCheckpointResponse]:
        """
        Start loading a checkpoint from a directory (async operation).

        This method sends the request immediately and returns a future.
        Use wait_until_ready() to poll until the checkpoint is fully loaded.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            TinkerbellFuture[LoadCheckpointResponse] - call .result() to poll for the response
        """
        # Send request immediately
        request = LoadCheckpointRequest(
            model_id=self.model_id,
            checkpoint_path=checkpoint_path,
            pin_lora=pin_lora,
        )

        response = self.client.post(
            "/load_checkpoint",
            json=request.model_dump(exclude_none=True),
        )

        response.raise_for_status()
        remote_future = response.json()

        # Parse result
        def _parse_result(result: dict[str, Any]) -> LoadCheckpointResponse:
            return LoadCheckpointResponse(**result)
        
        return TinkerbellFuture(
            request_id=remote_future["request_id"],
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=_parse_result,
            poll_interval=0.1,
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
        remote_future = response.json()
        
        # Parse result
        def _parse_result(result: dict[str, Any]) -> ShutdownSamplingActorResponse:
            return ShutdownSamplingActorResponse(**result)
        
        return TinkerbellFuture(
            request_id=remote_future["request_id"],
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=_parse_result,
            poll_interval=0.1,
            timeout=self.timeout,
        )

    def close(self):
        """Close the HTTP client."""
        self.client.close()
