import time
from typing import Any, Optional

import httpx
from transformers import AutoTokenizer

from tinkerbell.client.sampling import SamplingClient
from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.requests import (
    ActorStatusRequest,
    CreateSamplingActorRequest,
    ForwardRequest,
    SaveCheckpointRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    CreateSamplingActorResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    SaveCheckpointResponse,
)


class HuggingFaceTokenizer:
    def __init__(self, model_id: str):
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)

    def apply_chat_template(
        self, messages: list[dict[str, str]], *args, **kwargs
    ) -> list[str]:
        return self.hf_tokenizer.apply_chat_template(
            messages, tokenize=False, *args, **kwargs
        )

    def encode(self, *args, **kwargs) -> dict[str, TensorData]:
        kwargs["return_tensors"] = "pt"
        encoded = self.hf_tokenizer.encode(*args, **kwargs)
        return {k: TensorData.from_torch(v) for k, v in encoded.items()}

    def __call__(self, *args, **kwargs) -> dict[str, TensorData]:
        kwargs["return_tensors"] = "pt"
        output = self.hf_tokenizer(*args, **kwargs)
        return {k: TensorData.from_torch(v) for k, v in output.items()}


class TrainingClient:
    """Client for interacting with the Tinkerbell training service."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        timeout: float = 600.0,
    ):
        """
        Initialize the training client.

        Args:
            server_url: Server URL of the training service
            timeout: Request timeout in seconds
            tokenizer: Optional pre-loaded tokenizer
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
        self.tokenizer = self.get_tokenizer()

    def get_actor_status(self) -> ActorStatusResponse:
        """
        Get the status of training actors.

        Args:
        Returns:
            ActorStatusResponse with current status
        """
        request = ActorStatusRequest(model_id=self.model_id)
        response = self.client.post(
            "/get_actor_status",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    def wait_until_ready(
        self,
        poll_interval: float = 2.0,
        verbose: bool = True,
    ) -> None:
        """
        Poll the server until actors are ready.

        Args:
            poll_interval: Time between status checks in seconds
            verbose: If True, prints status updates
        """
        while True:
            status = self.get_actor_status()
            if verbose:
                print(f"Actor status: {status.status}")

            if status.status == "ready":
                if verbose:
                    print("Actors are ready!")
                break

            time.sleep(poll_interval)

    def get_tokenizer(self) -> HuggingFaceTokenizer:
        """
        Load and cache a tokenizer.

        Args:
        Returns:
            HuggingFaceTokenizer instance
        """
        self.tokenizer = HuggingFaceTokenizer(self.model_id)
        return self.tokenizer

    def zero_grad(self) -> dict[str, Any]:
        """
        Zero out gradients.

        Args:
        Returns:
            Response dictionary
        """
        response = self.client.post(
            "/zero_grad",
            json={"model_id": self.model_id},
        )
        response.raise_for_status()
        return response.json()

    def forward(
        self,
        inputs: dict[str, TensorData | Any],
        forward_kwargs: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ForwardResponse:
        """
        Perform forward pass.

        Args:
            model_id: ID of the model
            inputs: Input tensors (as dict of TensorData or serialized)
            forward_kwargs: Additional forward pass kwargs
            request_id: Optional request ID for async requests

        Returns:
            ForwardResponse
        """
        request = ForwardRequest(
            model_id=self.model_id,
            inputs=inputs,
            forward_kwargs=forward_kwargs or {},
            request_id=request_id,
        )

        response = self.client.post(
            "/forward",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ForwardResponse(**response.json())

    def forward_backward(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
    ) -> ForwardBackwardResponse:
        """
        Perform forward and backward pass.

        Args:
            data: List of Datum objects containing model inputs and loss function inputs
            forward_kwargs: Additional forward pass kwargs
            return_logprobs: If True, returns log probabilities

        Returns:
            ForwardBackwardResponse with request_id for async retrieval
        """
        request_data = {
            "model_id": self.model_id,
            "data": [datum.model_dump() for datum in data],
            "forward_kwargs": forward_kwargs or {},
            "return_logprobs": return_logprobs,
        }

        response = self.client.post(
            "/forward_backward",
            json=request_data,
        )
        response.raise_for_status()
        return ForwardBackwardResponse(**response.json())

    def get_result(self, request_id: str) -> dict[str, Any]:
        """
        Get the result of an async forward/backward request.

        Args:
            request_id: Request ID from forward_backward call

        Returns:
            Dictionary with loss and other outputs (e.g., {'loss': [0.5, 0.6, ...]})
        """
        response = self.client.post(
            "/get_result",
            json={"request_id": request_id},
        )
        response.raise_for_status()
        return response.json()

    def optim_step(
        self,
        optimizer_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Perform optimizer step.

        Args:
            optimizer_params: Optional optimizer parameters

        Returns:
            Response dictionary
        """
        response = self.client.post(
            "/optim_step",
            json={
                "model_id": self.model_id,
                "optimizer_params": optimizer_params or {},
            },
        )
        response.raise_for_status()
        return response.json()

    def save_checkpoint(
        self,
        checkpoint_path: str,
    ) -> SaveCheckpointResponse:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path where checkpoint should be saved

        Returns:
            SaveCheckpointResponse
        """
        request = SaveCheckpointRequest(
            model_id=self.model_id,
            checkpoint_path=checkpoint_path,
        )

        response = self.client.post(
            "/save_checkpoint",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return SaveCheckpointResponse(**response.json())

    def save_weights_and_get_sampling_client(
        self,
        checkpoint_path: str,
        tp_size: Optional[int] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = True,
    ) -> "SamplingClient":
        """
        Save model weights and return a loaded sampling client for generation.

        Args:
            checkpoint_path: Path where checkpoint should be saved
            tp_size: Tensor parallel size for sampling (defaults to 1)
            engine_kwargs: SGLang engine configuration parameters
            wait_until_ready: Whether to block until sampling actor is ready

        Returns:
            SamplingClient instance ready for sampling
        """
        from tinkerbell.client.sampling import SamplingClient

        # Step 1: Save the checkpoint
        self.save_checkpoint(checkpoint_path)

        # Step 2: Create sampling actor
        request = CreateSamplingActorRequest(
            model_id=self.model_id,
            tp_size=tp_size or 1,
            engine_kwargs=engine_kwargs or {},
        )

        response = self.client.post(
            "/create_sampling_actor",
            json=request.model_dump(),
        )
        response.raise_for_status()
        result = CreateSamplingActorResponse(**response.json())

        if not result.success:
            raise RuntimeError(f"Failed to create sampling actor: {result.message}")

        # Step 3: Create SamplingClient
        sampling_client = SamplingClient(
            server_url=self.server_url,
            model_id=self.model_id,
            timeout=self.timeout,
        )
        _ = sampling_client.load_checkpoint(checkpoint_path)

        # Step 4: Wait until ready if requested
        if wait_until_ready:
            sampling_client.wait_until_ready()

        return sampling_client

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
