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
    ) -> None:
        """
        Poll the server until the inference actor is ready.

        Args:
            poll_interval: Time between status checks in seconds
            verbose: If True, prints status updates
        """
        while True:
            status = self.get_status()
            if verbose:
                print(f"Inference actor status: {status.status}")

            if status.status == "ready":
                if verbose:
                    print("Inference actor is ready!")
                break

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
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict = {},
    ) -> list[str]:
        """
        Generate text from prompts.

        Args:
            prompts: List of prompts to generate from
            sampling_params: Sampling parameters

        Returns:
            List of generated text strings
        """
        request = GenerateRequest(
            model_id=self.model_id,
            prompts=prompts,
            sampling_params=sampling_params,
        )

        response = self.client.post(
            "/generate",
            json=request.model_dump(),
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
            json=request.model_dump(),
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
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ShutdownInferenceActorResponse(**response.json())

    def close(self):
        """Close the HTTP client."""
        self.client.close()
