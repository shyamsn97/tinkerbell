import time

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


class SamplingClient:
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
        self.server_url = server_url
        self.timeout = timeout

        # Configure timeout with separate values for connect and read
        # This helps with VPN/proxy environments
        timeout_config = httpx.Timeout(
            connect=30.0,  # Connection timeout
            read=self.timeout,  # Read timeout
            write=30.0,  # Write timeout
            pool=30.0,  # Pool timeout
        )

        # Configure transport with retries and connection limits
        # Set higher limits to support concurrent requests from multiple threads
        limits = httpx.Limits(
            max_connections=200,  # Total connection pool size
            max_keepalive_connections=100,  # Keep-alive connections
        )

        transport = httpx.HTTPTransport(
            retries=3,  # Retry failed connections
            limits=limits,
        )

        self.client = httpx.Client(
            base_url=self.server_url,
            timeout=timeout_config,
            transport=transport,
            follow_redirects=True,
        )
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

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout:
                raise TimeoutError(
                    f"Sampling actor did not become ready within {timeout}s. "
                    f"Last status: {last_status}"
                )

            status = self.get_status()

            # Print if status changed or every 30 seconds
            status_changed = status.status != last_status
            should_print_interval = (time.time() - last_print_time) >= 30

            if verbose and (status_changed or should_print_interval):
                print(f"[{elapsed:.1f}s] Sampling actor status: {status.status}")
                last_print_time = time.time()

            last_status = status.status

            if status.status == "ready":
                if verbose:
                    print(f"✓ Sampling actor is ready! (took {elapsed:.1f}s)")
                break
            elif status.status == "not_present":
                raise RuntimeError(
                    f"Sampling actor became 'not_present' after {elapsed:.1f}s. "
                    f"This usually means the actor crashed during initialization or checkpoint loading. "
                    f"Check the Ray logs with: ray logs <actor_name>"
                )

            time.sleep(poll_interval)

    def get_status(self) -> ActorStatusResponse:
        """
        Get the status of the sampling actor.

        Returns:
            ActorStatusResponse with current status
        """
        request = ActorStatusRequest(model_id=self.model_id)
        response = self.client.post(
            "/get_sampling_actor_status",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    def sample(
        self,
        *args,
        **kwargs,
    ) -> SampleResponse:
        """
        Sample text from the request.

        Args:
            *args: Positional arguments to pass to the SampleRequest
            **kwargs: Keyword arguments to pass to the SampleRequest

        Returns:
            SampleResponse with outputs, logprobs, and other metadata
        """
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
        result = SampleResponse(**response.json())
        return result

    def load_checkpoint(
        self,
        checkpoint_path: str,
    ) -> LoadCheckpointResponse:
        """
        Start loading a checkpoint from a directory (async operation).

        This method returns immediately after starting the load operation.
        Use wait_until_ready() to poll until the checkpoint is fully loaded.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            LoadCheckpointResponse with success status and message
        """
        request = LoadCheckpointRequest(
            model_id=self.model_id,
            checkpoint_path=checkpoint_path,
        )

        response = self.client.post(
            "/load_checkpoint",
            json=request.model_dump(exclude_none=True),
        )

        response.raise_for_status()
        return LoadCheckpointResponse(**response.json())

    def shutdown(self) -> ShutdownSamplingActorResponse:
        """
        Shutdown the sampling actor.

        Returns:
            ShutdownSamplingActorResponse with success status and message
        """
        request = ShutdownSamplingActorRequest(model_id=self.model_id)

        response = self.client.post(
            "/shutdown_sampling_actor",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ShutdownSamplingActorResponse(**response.json())

    def close(self):
        """Close the HTTP client."""
        self.client.close()
