import logging
import time
from typing import Any, Optional

from tinkerbell.client.base import AsyncTinkerbellFuture, BaseClient, TinkerbellFuture
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.optimizer import OptimStepRequest, ZeroGradRequest
from tinkerbell.types.requests import (
    ActorStatusRequest,
    CreateSamplingActorRequest,
    ForwardRequest,
    PushToHubRequest,
    SaveCheckpointRequest,
)
from tinkerbell.types.responses import (
    ActorStatusResponse,
    CreateSamplingActorResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    PushToHubResponse,
    RemoteFuture,
    SaveCheckpointResponse,
)

logger = logging.getLogger(__name__)


class TrainingClient(BaseClient):
    """Client for interacting with the Tinkerbell training service."""

    def __init__(
        self,
        server_url: str,
        model_id: str,
        model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        timeout: float = 600.0,
        lora_enabled: bool = False,
        lora_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(server_url, timeout)
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.adapter_name = adapter_name
        self.lora_enabled = lora_enabled
        self.lora_config = lora_config
        self._tokenizer = None

    def get_tokenizer(self):
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or 0
        return self._tokenizer

    def wait_until_ready(
        self, poll_interval: float = 2.0, verbose: bool = True
    ) -> None:
        """Poll the server until actors are ready."""
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
        """Get the status of training actors."""
        request = ActorStatusRequest(model_id=self.model_name)
        response = self.client.post(
            "/get_actor_status", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    # ===== Training operations =====

    def zero_grad(self) -> TinkerbellFuture[dict[str, Any]]:
        """Zero out gradients."""
        return self.create_future(
            request=ZeroGradRequest(
                model_id=self.model_name, adapter_name=self.adapter_name
            ),
            endpoint="/zero_grad",
        )

    async def zero_grad_async(self) -> AsyncTinkerbellFuture[dict[str, Any]]:
        """Async: Zero out gradients."""
        request = ZeroGradRequest(
            model_id=self.model_name, adapter_name=self.adapter_name
        )
        response = await self.async_client.post(
            "/zero_grad", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(**response.json()),
            server_url=self.server_url,
            result_parser=lambda x: x,
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def _build_forward_request(
        self,
        inputs: dict[str, TensorData | Any],
        forward_kwargs: Optional[dict[str, Any]],
        request_id: Optional[str],
    ) -> ForwardRequest:
        """Build forward request."""
        return ForwardRequest(
            model_id=self.model_name,
            inputs=inputs,
            forward_kwargs=forward_kwargs or {},
            request_id=request_id,
        )

    def _parse_forward_response(
        self, initial: ForwardResponse, result: dict[str, Any]
    ) -> ForwardResponse:
        """Parse forward result into response object."""
        return ForwardResponse(
            model_id=initial.model_id,
            request_id=initial.request_id,
            logprobs=result.get("logprobs"),
            outputs=result.get("outputs"),
            metrics=result.get("metrics"),
        )

    def forward(
        self,
        inputs: dict[str, TensorData | Any],
        forward_kwargs: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> TinkerbellFuture[ForwardResponse]:
        """Perform forward pass."""
        request = self._build_forward_request(inputs, forward_kwargs, request_id)
        response = self.client.post("/forward", json=request.model_dump())
        response.raise_for_status()
        initial = ForwardResponse(**response.json())

        parse = lambda result: self._parse_forward_response(initial, result)
        return self.create_future_from_request_id(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_id=initial.model_id
            ),
            parse_result_fn=parse,
        )

    async def forward_async(
        self,
        inputs: dict[str, TensorData | Any],
        forward_kwargs: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> AsyncTinkerbellFuture[ForwardResponse]:
        """Async: Perform forward pass. First await submits request, second await gets result."""
        request = self._build_forward_request(inputs, forward_kwargs, request_id)
        response = await self.async_client.post("/forward", json=request.model_dump())
        response.raise_for_status()
        initial = ForwardResponse(**response.json())

        parse = lambda result: self._parse_forward_response(initial, result)
        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_id=initial.model_id
            ),
            server_url=self.server_url,
            result_parser=parse,
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def _build_forward_backward_request(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]],
        return_logprobs: bool,
    ) -> dict[str, Any]:
        """Build forward_backward request payload."""
        return {
            "model_id": self.model_name,
            "adapter_name": self.adapter_name,
            "data": [datum.model_dump() for datum in data],
            "forward_kwargs": forward_kwargs or {},
            "return_logprobs": return_logprobs,
        }

    def _parse_forward_backward_response(
        self, initial: ForwardBackwardResponse, result: dict[str, Any]
    ) -> ForwardBackwardResponse:
        """Parse forward_backward result into response object."""
        return ForwardBackwardResponse(
            model_id=initial.model_id,
            request_id=initial.request_id,
            loss=result.get("loss"),
            logprobs=result.get("logprobs"),
            outputs=result.get("outputs"),
            metrics=result.get("metrics"),
            sum_gradient=result.get("sum_gradient"),
        )

    def forward_backward(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
    ) -> TinkerbellFuture[ForwardBackwardResponse]:
        """Perform forward and backward pass."""
        request_data = self._build_forward_backward_request(
            data, forward_kwargs, return_logprobs
        )
        response = self.client.post("/forward_backward", json=request_data)
        response.raise_for_status()
        initial = ForwardBackwardResponse(**response.json())

        parse = lambda result: self._parse_forward_backward_response(initial, result)
        return self.create_future_from_request_id(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_id=initial.model_id
            ),
            parse_result_fn=parse,
        )

    async def forward_backward_async(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
    ) -> AsyncTinkerbellFuture[ForwardBackwardResponse]:
        """Async: Perform forward and backward pass. First await submits request, second await gets result."""
        request_data = self._build_forward_backward_request(
            data, forward_kwargs, return_logprobs
        )
        response = await self.async_client.post("/forward_backward", json=request_data)
        response.raise_for_status()
        initial = ForwardBackwardResponse(**response.json())

        parse = lambda result: self._parse_forward_backward_response(initial, result)
        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_id=initial.model_id
            ),
            server_url=self.server_url,
            result_parser=parse,
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def get_result(self, request_id: str) -> TinkerbellFuture[dict[str, Any]]:
        """Get the result of an async forward/backward request (deprecated)."""
        return self.create_future_from_request_id(
            RemoteFuture(request_id=request_id, model_id=self.model_name)
        )

    def optim_step(
        self, optimizer_params: Optional[dict[str, Any]] = None
    ) -> TinkerbellFuture[dict[str, Any]]:
        """Perform optimizer step for this client's adapter."""
        request = OptimStepRequest(
            model_id=self.model_name,
            adapter_name=self.adapter_name,
            optimizer_params=optimizer_params or {},
        )
        return self.create_future(request=request, endpoint="/optim_step")

    async def optim_step_async(
        self, optimizer_params: Optional[dict[str, Any]] = None
    ) -> AsyncTinkerbellFuture[dict[str, Any]]:
        """Async: Perform optimizer step for this client's adapter."""
        request = OptimStepRequest(
            model_id=self.model_name,
            adapter_name=self.adapter_name,
            optimizer_params=optimizer_params or {},
        )
        response = await self.async_client.post(
            "/optim_step", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(**response.json()),
            server_url=self.server_url,
            result_parser=lambda x: x,
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def save_checkpoint(
        self, checkpoint_path: str
    ) -> TinkerbellFuture[SaveCheckpointResponse]:
        """Save model checkpoint (or specific adapter if lora_enabled)."""
        request = SaveCheckpointRequest(
            model_id=self.model_name,
            checkpoint_path=checkpoint_path,
            adapter_name=self.adapter_name,
        )
        return self.create_future(
            request=request,
            endpoint="/save_checkpoint",
            parse_result_fn=lambda r: SaveCheckpointResponse(**r),
        )

    async def save_checkpoint_async(
        self, checkpoint_path: str
    ) -> AsyncTinkerbellFuture[SaveCheckpointResponse]:
        """Async: Save model checkpoint."""
        request = SaveCheckpointRequest(
            model_id=self.model_name,
            checkpoint_path=checkpoint_path,
            adapter_name=self.adapter_name,
        )
        response = await self.async_client.post(
            "/save_checkpoint", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(**response.json()),
            server_url=self.server_url,
            result_parser=lambda r: SaveCheckpointResponse(**r),
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        push_kwargs: Optional[dict[str, Any]] = None,
    ) -> TinkerbellFuture[PushToHubResponse]:
        """Push model to Hugging Face Hub (or specific adapter if lora_enabled).

        The repository will be created automatically if it doesn't exist.
        Requires authentication via token or huggingface_hub login.
        """
        logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        request = PushToHubRequest(
            model_id=self.model_name,
            repo_id=repo_id,
            adapter_name=self.adapter_name,
            token=token,
            private=private,
            commit_message=commit_message,
            push_kwargs=push_kwargs or {},
        )
        return self.create_future(
            request=request,
            endpoint="/push_to_hub",
            parse_result_fn=lambda r: PushToHubResponse(**r),
        )

    async def push_to_hub_async(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        push_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncTinkerbellFuture[PushToHubResponse]:
        """Async: Push model to Hugging Face Hub.

        The repository will be created automatically if it doesn't exist.
        Requires authentication via token or huggingface_hub login.
        """
        logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        request = PushToHubRequest(
            model_id=self.model_name,
            repo_id=repo_id,
            adapter_name=self.adapter_name,
            token=token,
            private=private,
            commit_message=commit_message,
            push_kwargs=push_kwargs or {},
        )
        response = await self.async_client.post(
            "/push_to_hub", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return AsyncTinkerbellFuture(
            remote_future=RemoteFuture(**response.json()),
            server_url=self.server_url,
            result_parser=lambda r: PushToHubResponse(**r),
            poll_interval=1.0,
            timeout=self.timeout,
        )

    def save_weights_and_get_sampling_client(
        self,
        checkpoint_path: str,
        tp_size: Optional[int] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = False,
    ) -> SamplingClient:
        """Save weights and return a sampling client.

        For LoRA: Creates actor with base model, then loads LoRA adapter.
        For full model: Creates actor directly with checkpoint path (SGLang does conversion).
        """
        from tinkerbell.client.sampling import SamplingClient

        self.save_checkpoint(checkpoint_path).result()

        is_lora = self.adapter_name is not None
        final_engine_kwargs = engine_kwargs or {}

        if is_lora:
            # LoRA: Create actor with base model, then load adapter
            actor_model_id = self.model_id
        else:
            # Full model: Create actor directly with checkpoint (SGLang converts HF format)
            # Disable LoRA mode since it's not needed
            actor_model_id = checkpoint_path
            final_engine_kwargs = {**final_engine_kwargs, "enable_lora": False}

        request = CreateSamplingActorRequest(
            model_id=actor_model_id,
            model_name=self.model_name,
            tp_size=tp_size or 1,
            engine_kwargs=final_engine_kwargs,
        )
        response = self.client.post("/create_sampling_actor", json=request.model_dump())
        response.raise_for_status()
        result = CreateSamplingActorResponse(**response.json())
        if not result.success:
            raise RuntimeError(f"Failed to create sampling actor: {result.message}")

        sampling_client = SamplingClient(
            server_url=self.server_url,
            model_id=self.model_name,
            adapter_name=self.adapter_name,
            timeout=self.timeout,
        )

        if wait_until_ready:
            sampling_client.wait_until_ready()

        # Only load checkpoint for LoRA (full model already loaded from checkpoint path)
        if is_lora:
            sampling_client.load_checkpoint(checkpoint_path).result()

        return sampling_client

    # ===== Cleanup =====

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
