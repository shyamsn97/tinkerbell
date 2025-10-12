# """HTTP client for connecting to the FastAPI PyTorch training server."""

# from typing import Any, Optional

# import httpx
# from transformers.tokenization_utils import PreTrainedTokenizer

# from tinkerbell.client.training_client import TrainingClient
# from tinkerbell.utils import deserialize_tensor, serialize_payload, serialize_tensor


# class TorchTrainingClient(TrainingClient):
#     """HTTP client for communicating with TorchTrainingServer."""

#     def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
#         """Initialize the torch training client.

#         Args:
#             base_url: Base URL of the training server
#             timeout: Request timeout in seconds
#         """
#         self.base_url = base_url.rstrip("/")
#         self.timeout = timeout
#         self.client = httpx.Client(timeout=timeout)
#         self._tokenizer = None

#     def __enter__(self):
#         """Context manager entry."""
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Context manager exit."""
#         self.close()

#     def close(self):
#         """Close the HTTP client."""
#         self.client.close()

#     def _post(self, endpoint: str, data: bytes) -> Any:
#         """Make a POST request to the server.

#         Args:
#             endpoint: API endpoint (e.g., '/forward')
#             data: Serialized bytes to send

#         Returns:
#             Deserialized response
#         """
#         url = f"{self.base_url}{endpoint}"
#         response = self.client.post(
#             url, content=data, headers={"Content-Type": "application/octet-stream"}
#         )
#         response.raise_for_status()
#         return deserialize_tensor(response.content)

#     def _get(self, endpoint: str) -> Any:
#         """Make a GET request to the server.

#         Args:
#             endpoint: API endpoint (e.g., '/health')

#         Returns:
#             Deserialized response or JSON
#         """
#         url = f"{self.base_url}{endpoint}"
#         response = self.client.get(url)
#         response.raise_for_status()

#         # Try to parse as JSON first
#         content_type = response.headers.get("content-type", "")
#         if "application/json" in content_type:
#             return response.json()
#         else:
#             return deserialize_tensor(response.content)

#     def forward(self, data: list[Any], loss_fn: Optional[Any] = None) -> Any:
#         """Forward pass through the model and compute the loss.

#         Args:
#             data: A list of data points to be processed by the model
#             loss_fn: A loss function to compute the loss

#         Returns:
#             Loss values or model outputs
#         """
#         payload = serialize_payload(data, loss_fn)
#         return self._post("/forward", payload)

#     def forward_backward(self, data: list[Any], loss_fn: Optional[Any] = None) -> Any:
#         """Forward and backward pass through the model and compute the loss.

#         Args:
#             data: A list of data points to be processed by the model
#             loss_fn: A loss function to compute the loss

#         Returns:
#             Loss values
#         """
#         payload = serialize_payload(data, loss_fn)
#         return self._post("/forward_backward", payload)

#     def optim_step(self, optimizer_params: Optional[dict] = None) -> Any:
#         """Perform an optimization step.

#         Args:
#             optimizer_params: Optional optimizer parameters

#         Returns:
#             Optimizer status
#         """
#         if optimizer_params is None:
#             optimizer_params = {}
#         data = serialize_tensor(optimizer_params)
#         return self._post("/optim_step", data)

#     def save_state(self) -> dict:
#         """Save the state of the model.

#         Returns:
#             Dictionary containing model state
#         """
#         return self._get("/save_state")

#     def load_state(self, state: dict) -> dict:
#         """Load the state of the model.

#         Args:
#             state: Dictionary containing model state

#         Returns:
#             Status dictionary
#         """
#         data = serialize_tensor(state)
#         return self._post("/load_state", data)

#     def health_check(self) -> dict:
#         """Check if the server is healthy.

#         Returns:
#             Health status dictionary
#         """
#         return self._get("/health")

#     def get_tokenizer(self) -> PreTrainedTokenizer:
#         """Get the tokenizer for the model.

#         Note: This is a placeholder implementation. In a real scenario,
#         you might want to add an endpoint to the server to retrieve the tokenizer,
#         or pass it during client initialization.

#         Returns:
#             A tokenizer (currently returns stored tokenizer)
#         """
#         if self._tokenizer is None:
#             raise NotImplementedError(
#                 "Tokenizer must be set using set_tokenizer() method or "
#                 "retrieved from the server via a dedicated endpoint."
#             )
#         return self._tokenizer

#     def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
#         """Set the tokenizer for the client.

#         Args:
#             tokenizer: HuggingFace tokenizer to use
#         """
#         self._tokenizer = tokenizer

#     def create_sampling_client(self) -> Any:
#         """Create a sampling client for the model.

#         Note: This is a placeholder. Implement based on your specific needs.

#         Returns:
#             A sampling client
#         """
#         raise NotImplementedError(
#             "Sampling client creation not yet implemented. "
#             "This should be customized based on your use case."
#         )

#     def deploy(self, config: Optional[dict] = None) -> dict:
#         """Deploy configuration (client-side is no-op).

#         Args:
#             config: Optional configuration dictionary

#         Returns:
#             Configuration dictionary
#         """
#         return config or {}
