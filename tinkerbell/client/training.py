import time
from typing import Any, Optional

import httpx
from transformers import AutoTokenizer

from tinkerbell.types.data import TensorData
from tinkerbell.types.requests import (
    ActorStatusRequest,
    CreateTrainingActorsRequest,
    ForwardRequest,
    SaveCheckpointRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    CreateTrainingActorsResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    SaveCheckpointResponse,
)


class TrainingClient:
    """Client for interacting with the Tinkerbell training service."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize the training client.

        Args:
            base_url: Base URL of the training service
            timeout: Request timeout in seconds
            tokenizer: Optional pre-loaded tokenizer
        """
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self.tokenizer = tokenizer

    def create_training_actors(
        self,
        model_name: str,
        tp_size: int,
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
        rank: int = 0,
        parallelize_plan: Optional[dict[str, str]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        scheduler_params: Optional[dict[str, Any]] = None,
        lora_config: Optional[dict[str, Any]] = None,
        ray_worker_options: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = False,
    ) -> CreateTrainingActorsResponse:
        """
        Create training actors on the server.

        Args:
            model_name: Name or path of the model
            world_size: Number of processes in distributed training
            master_addr: Address of the master process
            master_port: Port of the master process
            rank: Rank of this process
            parallelize_plan: Dictionary mapping layer patterns to parallelization strategy
            model_kwargs: Additional kwargs for model initialization
            scheduler_params: Learning rate scheduler parameters
            lora_config: LoRA configuration
            ray_worker_options: Ray worker options
            wait_until_ready: If True, blocks until actors are ready

        Returns:
            CreateTrainingActorsResponse with status
        """
        request = CreateTrainingActorsRequest(
            model_name=model_name,
            world_size=tp_size,
            master_addr=master_addr,
            master_port=master_port,
            rank=rank,
            parallelize_plan=parallelize_plan or {},
            model_kwargs=model_kwargs or {},
            scheduler_params=scheduler_params or {},
            lora_config=lora_config or {},
            ray_worker_options=ray_worker_options or {},
            wait_until_ready=wait_until_ready,
        )

        response = self.client.post(
            "/create_training_actors",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return CreateTrainingActorsResponse(**response.json())

    def get_actor_status(self, model_name: str) -> ActorStatusResponse:
        """
        Get the status of training actors.

        Args:
            model_name: Name of the model

        Returns:
            ActorStatusResponse with current status
        """
        request = ActorStatusRequest(model_name=model_name)
        response = self.client.post(
            "/get_actor_status",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    def wait_until_ready(
        self,
        model_name: str,
        poll_interval: float = 2.0,
        verbose: bool = True,
    ) -> None:
        """
        Poll the server until actors are ready.

        Args:
            model_name: Name of the model
            poll_interval: Time between status checks in seconds
            verbose: If True, prints status updates
        """
        while True:
            status = self.get_actor_status(model_name)
            if verbose:
                print(f"Actor status: {status.status}")

            if status.status == "ready":
                if verbose:
                    print("Actors are ready!")
                break

            time.sleep(poll_interval)

    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        Load and cache a tokenizer.

        Args:
            model_name: Name or path of the model

        Returns:
            AutoTokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def tokenize(
        self,
        texts: list[str],
        tokenizer: Optional[AutoTokenizer] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        add_labels: bool = True,
    ) -> dict[str, TensorData]:
        """
        Tokenize input texts.

        Args:
            texts: List of text strings to tokenize
            tokenizer: Tokenizer to use (defaults to self.tokenizer)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            add_labels: If True, adds labels field (copy of input_ids)

        Returns:
            Dictionary with input_ids, attention_mask, and optionally labels
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("No tokenizer available. Call load_tokenizer() first.")

        encoded = tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {
            "input_ids": TensorData.from_torch(encoded["input_ids"]),
            "attention_mask": TensorData.from_torch(encoded["attention_mask"]),
        }

        if add_labels:
            inputs["labels"] = TensorData.from_torch(encoded["input_ids"])

        return inputs

    def zero_grad(self, model_name: str) -> dict[str, Any]:
        """
        Zero out gradients.

        Args:
            model_name: Name of the model

        Returns:
            Response dictionary
        """
        response = self.client.post(
            "/zero_grad",
            json={"model_name": model_name},
        )
        response.raise_for_status()
        return response.json()

    def forward(
        self,
        model_name: str,
        inputs: dict[str, Any],
        forward_kwargs: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> ForwardResponse:
        """
        Perform forward pass.

        Args:
            model_name: Name of the model
            inputs: Input tensors (as dict of TensorData or serialized)
            forward_kwargs: Additional forward pass kwargs
            request_id: Optional request ID for async requests

        Returns:
            ForwardResponse
        """
        request = ForwardRequest(
            model_name=model_name,
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
        model_name: str,
        inputs: dict[str, Any],
        targets: Optional[Any] = None,
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
        request_id: Optional[str] = None,
    ) -> ForwardBackwardResponse:
        """
        Perform forward and backward pass.

        Args:
            model_name: Name of the model
            inputs: Input tensors (should NOT include labels)
            targets: Target tensors for loss calculation
            forward_kwargs: Additional forward pass kwargs
            return_logprobs: If True, returns log probabilities
            request_id: Optional request ID for async requests

        Returns:
            ForwardBackwardResponse with request_id for async retrieval
        """
        request_data = {
            "model_name": model_name,
            "inputs": inputs,
            "forward_kwargs": forward_kwargs or {},
            "return_logprobs": return_logprobs,
        }

        if targets is not None:
            request_data["targets"] = targets

        if request_id is not None:
            request_data["request_id"] = request_id

        response = self.client.post(
            "/forward_backward",
            json=request_data,
        )
        response.raise_for_status()
        return ForwardBackwardResponse(**response.json())

    def get_result(self, request_id: str) -> ForwardBackwardResponse:
        """
        Get the result of an async forward/backward request.

        Args:
            request_id: Request ID from forward_backward call

        Returns:
            ForwardBackwardResponse with loss and outputs
        """
        response = self.client.post(
            "/get_result",
            json={"request_id": request_id},
        )
        response.raise_for_status()
        return ForwardBackwardResponse(**response.json())

    def optim_step(
        self,
        model_name: str,
        optimizer_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Perform optimizer step.

        Args:
            model_name: Name of the model
            optimizer_params: Optional optimizer parameters

        Returns:
            Response dictionary
        """
        response = self.client.post(
            "/optim_step",
            json={
                "model_name": model_name,
                "optimizer_params": optimizer_params or {},
            },
        )
        response.raise_for_status()
        return response.json()

    def save_checkpoint(
        self,
        model_name: str,
        checkpoint_path: str,
    ) -> SaveCheckpointResponse:
        """
        Save model checkpoint.

        Args:
            model_name: Name of the model
            checkpoint_path: Path where checkpoint should be saved

        Returns:
            SaveCheckpointResponse
        """
        request = SaveCheckpointRequest(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )

        response = self.client.post(
            "/save_checkpoint",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return SaveCheckpointResponse(**response.json())

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
