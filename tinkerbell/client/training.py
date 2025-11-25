import logging
import time
from typing import Any, Optional

from tinkerbell.client.base import BaseClient, TinkerbellFuture
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.optimizer import OptimStepRequest
from tinkerbell.types.requests import (
    ActorStatusRequest,
    CreateSamplingActorRequest,
    ForwardRequest,
    SaveCheckpointRequest,
    ZeroGradRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    CreateSamplingActorResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    SaveCheckpointResponse,
)

logger = logging.getLogger(__name__)


class HuggingFaceTokenizer:
    def __init__(self, model_id: str):
        from transformers import AutoTokenizer

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


class TrainingClient(BaseClient):
    """Client for interacting with the Tinkerbell training service."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        timeout: float = 600.0,
        lora_enabled: bool = False,
        lora_config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the training client.

        Args:
            server_url: Server URL of the training service
            timeout: Request timeout in seconds
            lora_enabled: Whether LoRA was enabled for training
            lora_config: LoRA configuration dict
            tokenizer: Optional pre-loaded tokenizer
        """
        super().__init__(server_url, timeout)
        self.lora_enabled = lora_enabled
        self.lora_config = lora_config
        self.model_id = model_id
        self.tokenizer = self.get_tokenizer()

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
                logger.info(f"Actor status: {status.status}")

            if status.status == "ready":
                if verbose:
                    logger.info("Actors are ready!")
                break

            time.sleep(poll_interval)

    def get_actor_status(self) -> ActorStatusResponse:
        """
        Get the status of training actors.

        Returns:
            ActorStatusResponse - the status response (synchronous call)
        """
        request = ActorStatusRequest(model_id=self.model_id)

        response = self.client.post(
            "/get_actor_status",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    def get_tokenizer(self) -> HuggingFaceTokenizer:
        """
        Load and cache a tokenizer.

        Args:
        Returns:
            HuggingFaceTokenizer instance
        """
        self.tokenizer = HuggingFaceTokenizer(self.model_id)
        return self.tokenizer

    def zero_grad(self) -> TinkerbellFuture[dict[str, Any]]:
        """
        Zero out gradients.

        Returns:
            TinkerbellFuture[dict[str, Any]] - call .result() to poll for the response
        """
        request = ZeroGradRequest(model_id=self.model_id)

        return self.create_future(request=request, endpoint="/zero_grad")

    def forward(
        self,
        inputs: dict[str, TensorData | Any],
        forward_kwargs: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> TinkerbellFuture[ForwardResponse]:
        """
        Perform forward pass.

        Args:
            model_id: ID of the model
            inputs: Input tensors (as dict of TensorData or serialized)
            forward_kwargs: Additional forward pass kwargs
            request_id: Optional request ID for async requests

        Returns:
            TinkerbellFuture[ForwardResponse] - call .result() to get the actual response
        """
        # Make the initial request
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
        initial_response = ForwardResponse(**response.json())

        # Create future for polling
        def _parse_result(result: dict[str, Any]) -> ForwardResponse:
            return ForwardResponse(
                model_id=initial_response.model_id,
                request_id=initial_response.request_id,
                logprobs=result.get("logprobs"),
                outputs=result.get("outputs"),
                metrics=result.get("metrics"),
            )

        return self.create_future_from_request_id(
            request_id=initial_response.request_id,
            parse_result_fn=_parse_result,
        )

    def forward_backward(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
    ) -> TinkerbellFuture[ForwardBackwardResponse]:
        """
        Perform forward and backward pass.

        Args:
            data: List of Datum objects containing model inputs and loss function inputs
            forward_kwargs: Additional forward pass kwargs
            return_logprobs: If True, returns log probabilities

        Returns:
            TinkerbellFuture[ForwardBackwardResponse] - call .result() to get the actual response
        """
        # Make the initial request to start the operation
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
        initial_response = ForwardBackwardResponse(**response.json())

        # Create future with request_id for polling
        def _parse_result(result: dict[str, Any]) -> ForwardBackwardResponse:
            # Merge the initial response data with the polled result
            return ForwardBackwardResponse(
                model_id=initial_response.model_id,
                request_id=initial_response.request_id,
                loss=result.get("loss"),
                logprobs=result.get("logprobs"),
                outputs=result.get("outputs"),
                metrics=result.get("metrics"),
            )

        return self.create_future_from_request_id(
            request_id=initial_response.request_id,
            parse_result_fn=_parse_result,
        )

    def get_result(self, request_id: str) -> TinkerbellFuture[dict[str, Any]]:
        """
        Get the result of an async forward/backward request.

        Note: Deprecated - use the TinkerbellFuture returned by forward_backward() instead.
        This creates a future that polls for the given request_id.

        Args:
            request_id: Request ID from forward_backward call

        Returns:
            TinkerbellFuture[dict[str, Any]] - call .result() to poll for the response
        """
        return self.create_future_from_request_id(request_id=request_id)

    def optim_step(
        self,
        optimizer_params: Optional[dict[str, Any]] = None,
    ) -> TinkerbellFuture[dict[str, Any]]:
        """
        Perform optimizer step.

        Args:
            optimizer_params: Optional optimizer parameters

        Returns:
            TinkerbellFuture[dict[str, Any]] - call .result() to poll for the response
        """
        request = OptimStepRequest(
            model_id=self.model_id,
            optimizer_params=optimizer_params or {},
        )

        return self.create_future(request=request, endpoint="/optim_step")

    def save_checkpoint(
        self,
        checkpoint_path: str,
    ) -> TinkerbellFuture[SaveCheckpointResponse]:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path where checkpoint should be saved

        Returns:
            TinkerbellFuture[SaveCheckpointResponse] - call .result() to poll for the response
        """
        request = SaveCheckpointRequest(
            model_id=self.model_id,
            checkpoint_path=checkpoint_path,
        )

        def _parse_result(result: dict[str, Any]) -> SaveCheckpointResponse:
            return SaveCheckpointResponse(**result)

        return self.create_future(
            request=request,
            endpoint="/save_checkpoint",
            parse_result_fn=_parse_result,
        )

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
        self.save_checkpoint(checkpoint_path).result()

        # Step 2: Create sampling actor (LoRA defaults handled by SamplingActor)
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

        # Step 4: Wait until model is loaded
        if wait_until_ready:
            sampling_client.wait_until_ready()

        # Step 5: Load the checkpoint AFTER the model is initialized
        _ = sampling_client.load_checkpoint(checkpoint_path).result()

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
