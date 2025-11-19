import time

import httpx

from tinkerbell.types.requests import (
    ActorStatusRequest,
    GenerateRequest,
    LoadCheckpointRequest,
    ShutdownInferenceActorRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    GenerateResponse,
    LoadCheckpointResponse,
    ShutdownInferenceActorResponse,
)


class InferenceClient:
    """Client for interacting with the Tinkerbell inference service."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        timeout: float = 600.0,
    ):
        """
        Initialize the inference client.

        Args:
            server_url: Server URL of the inference service
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
        Poll the server until the inference actor is ready.

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
                    f"Inference actor did not become ready within {timeout}s. "
                    f"Last status: {last_status}"
                )

            status = self.get_status()

            # Print if status changed or every 30 seconds
            status_changed = status.status != last_status
            should_print_interval = (time.time() - last_print_time) >= 30

            if verbose and (status_changed or should_print_interval):
                print(f"[{elapsed:.1f}s] Inference actor status: {status.status}")
                last_print_time = time.time()

            last_status = status.status

            if status.status == "ready":
                if verbose:
                    print(f"✓ Inference actor is ready! (took {elapsed:.1f}s)")
                break
            elif status.status == "not_present":
                raise RuntimeError(
                    f"Inference actor became 'not_present' after {elapsed:.1f}s. "
                    f"This usually means the actor crashed during initialization or checkpoint loading. "
                    f"Check the Ray logs with: ray logs <actor_name>"
                )

            time.sleep(poll_interval)

    def get_status(self) -> ActorStatusResponse:
        """
        Get the status of the inference actor.

        Returns:
            ActorStatusResponse with current status
        """
        request = ActorStatusRequest(model_id=self.model_id)
        response = self.client.post(
            "/get_inference_actor_status",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    def generate(
        self,
        *args,
        **kwargs,
    ) -> list[str]:
        """
        Generate text from the request.

        Args:
            *args: Positional arguments to pass to the GenerateRequest
            **kwargs: Keyword arguments to pass to the GenerateRequest

        Returns:
            List of generated text strings
        """
        # Always include model_id from the client
        if "model_id" not in kwargs:
            kwargs["model_id"] = self.model_id

        request = GenerateRequest(
            *args,
            **kwargs,
        )

        response = self.client.post(
            "/generate",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        result = GenerateResponse(**response.json())
        return result.outputs

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

    def shutdown(self) -> ShutdownInferenceActorResponse:
        """
        Shutdown the inference actor.

        Returns:
            ShutdownInferenceActorResponse with success status and message
        """
        request = ShutdownInferenceActorRequest(model_id=self.model_id)

        response = self.client.post(
            "/shutdown_inference_actor",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ShutdownInferenceActorResponse(**response.json())

    def close(self):
        """Close the HTTP client."""
        self.client.close()
