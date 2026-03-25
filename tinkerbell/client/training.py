import logging
import time
from typing import Any, Optional

from tinkerbell.client.base import AsyncTinkerbellFuture, BaseClient, TinkerbellFuture
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.types import LossFnType
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
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


class TrainingClient(BaseClient):
    """Client for interacting with the Tinkerbell training service."""

    def __init__(
        self,
        server_url: str,
        base_model: str,
        model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        timeout: float = 600.0,
        lora_enabled: bool = False,
        lora_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(server_url=server_url, timeout=timeout, base_model=base_model)
        self.model_name = model_name if model_name else clean_model_name(base_model)
        self.adapter_name = adapter_name
        self.lora_enabled = lora_enabled
        self.lora_config = lora_config

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
        request = ActorStatusRequest(model_name=self.model_name)
        response = self.client.post(
            "/get_actor_status", json=request.model_dump(exclude_none=True)
        )
        response.raise_for_status()
        return ActorStatusResponse(**response.json())

    # ===== Training operations =====

    def _zero_grad_request(self, immediate: bool = False) -> ZeroGradRequest:
        return ZeroGradRequest(
            model_name=self.model_name,
            adapter_name=self.adapter_name,
            immediate=immediate,
        )

    def zero_grad(self, immediate: bool = False) -> TinkerbellFuture[dict[str, Any]]:
        """Zero out gradients.

        Args:
            immediate: If True, process the batch queue immediately instead of waiting for clock cycle.
        """
        return self.create_future(
            request=self._zero_grad_request(immediate), endpoint="/zero_grad"
        )

    async def zero_grad_async(
        self, immediate: bool = False
    ) -> AsyncTinkerbellFuture[dict[str, Any]]:
        """Async: Zero out gradients.

        Args:
            immediate: If True, process the batch queue immediately instead of waiting for clock cycle.
        """
        return await self.create_async_future(
            request=self._zero_grad_request(immediate), endpoint="/zero_grad"
        )

    def _build_forward_request(
        self,
        inputs: dict[str, TensorData | Any],
        forward_kwargs: Optional[dict[str, Any]],
        request_id: Optional[str],
    ) -> ForwardRequest:
        """Build forward request."""
        return ForwardRequest(
            model_name=self.model_name,
            inputs=inputs,
            forward_kwargs=forward_kwargs or {},
            request_id=request_id,
        )

    def _parse_forward_response(
        self, initial: ForwardResponse, result: dict[str, Any]
    ) -> ForwardResponse:
        """Parse forward result into response object."""
        return ForwardResponse(
            model_name=initial.model_name,
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
        return self.create_future_from_remote(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_name=initial.model_name
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

        return self.create_async_future_from_request_id(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_name=initial.model_name
            ),
            parse_result_fn=lambda result: self._parse_forward_response(
                initial, result
            ),
        )

    def _build_forward_backward_request(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]],
        return_logprobs: bool,
        zero_grad: bool,
        optimizer_params: Optional[dict[str, Any]],
        loss_fn: LossFnType = "cross_entropy",
        immediate: bool = False,
    ) -> dict[str, Any]:
        """Build forward_backward request payload."""
        payload = {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "data": [datum.model_dump() for datum in data],
            "forward_kwargs": forward_kwargs or {},
            "return_logprobs": return_logprobs,
            "zero_grad": zero_grad,
            "loss_fn": loss_fn,
            "immediate": immediate,
        }
        if optimizer_params is not None:
            payload["optimizer_params"] = optimizer_params
        return payload

    def _parse_forward_backward_response(
        self, initial: ForwardBackwardResponse, result: dict[str, Any]
    ) -> ForwardBackwardResponse:
        """Parse forward_backward result into response object."""
        return ForwardBackwardResponse(
            model_name=initial.model_name,
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
        zero_grad: bool = True,
        optimizer_params: Optional[dict[str, Any]] = None,
        loss_fn: LossFnType = "cross_entropy",
        immediate: bool = False,
    ) -> TinkerbellFuture[ForwardBackwardResponse]:
        """Perform forward and backward pass, optionally with optimizer step.

        Args:
            data: List of Datum objects to process
            forward_kwargs: Additional kwargs for forward pass
            return_logprobs: Whether to return logprobs
            zero_grad: Whether to zero gradients before forward/backward (default True).
                      Set to False for gradient accumulation.
            optimizer_params: If provided, run optim_step after backward (single round trip).
                            Example: {"name": "adam", "lr": 1e-4}
            loss_fn: Loss function to use. One of "cross_entropy", "importance_sampling", "ppo".
            immediate: If True, process the batch queue immediately instead of waiting for clock cycle.
        """
        request_data = self._build_forward_backward_request(
            data,
            forward_kwargs,
            return_logprobs,
            zero_grad,
            optimizer_params,
            loss_fn,
            immediate,
        )
        response = self.client.post("/forward_backward", json=request_data)
        response.raise_for_status()
        initial = ForwardBackwardResponse(**response.json())

        parse = lambda result: self._parse_forward_backward_response(initial, result)
        return self.create_future_from_remote(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_name=initial.model_name
            ),
            parse_result_fn=parse,
        )

    async def forward_backward_async(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
        zero_grad: bool = True,
        optimizer_params: Optional[dict[str, Any]] = None,
        loss_fn: LossFnType = "cross_entropy",
        immediate: bool = False,
    ) -> AsyncTinkerbellFuture[ForwardBackwardResponse]:
        """Async: Perform forward and backward pass, optionally with optimizer step.

        Args:
            data: List of Datum objects to process
            forward_kwargs: Additional kwargs for forward pass
            return_logprobs: Whether to return logprobs
            zero_grad: Whether to zero gradients before forward/backward (default True).
                      Set to False for gradient accumulation.
            optimizer_params: If provided, run optim_step after backward (single round trip).
            loss_fn: Loss function to use. One of "cross_entropy", "importance_sampling", "ppo".
            immediate: If True, process the batch queue immediately instead of waiting for clock cycle.
        """
        request_data = self._build_forward_backward_request(
            data,
            forward_kwargs,
            return_logprobs,
            zero_grad,
            optimizer_params,
            loss_fn,
            immediate,
        )
        response = await self.async_client.post("/forward_backward", json=request_data)
        response.raise_for_status()
        initial = ForwardBackwardResponse(**response.json())

        return self.create_async_future_from_request_id(
            remote_future=RemoteFuture(
                request_id=initial.request_id, model_name=initial.model_name
            ),
            parse_result_fn=lambda result: self._parse_forward_backward_response(
                initial, result
            ),
        )

    def get_result(self, request_id: str) -> TinkerbellFuture[dict[str, Any]]:
        """Get the result of an async forward/backward request (deprecated)."""
        return self.create_future_from_remote(
            RemoteFuture(request_id=request_id, model_name=self.model_name)
        )

    def _optim_step_request(
        self,
        optimizer_params: Optional[dict[str, Any]] = None,
        immediate: bool = False,
    ) -> OptimStepRequest:
        return OptimStepRequest(
            model_name=self.model_name,
            adapter_name=self.adapter_name,
            optimizer_params=optimizer_params or {},
            immediate=immediate,
        )

    def optim_step(
        self,
        optimizer_params: Optional[dict[str, Any]] = None,
        immediate: bool = False,
    ) -> TinkerbellFuture[dict[str, Any]]:
        """Perform optimizer step for this client's adapter.

        Args:
            optimizer_params: Optimizer configuration (e.g. {"name": "adam", "lr": 1e-4})
            immediate: If True, process the batch queue immediately instead of waiting for clock cycle.
        """
        return self.create_future(
            request=self._optim_step_request(optimizer_params, immediate),
            endpoint="/optim_step",
        )

    async def optim_step_async(
        self,
        optimizer_params: Optional[dict[str, Any]] = None,
        immediate: bool = False,
    ) -> AsyncTinkerbellFuture[dict[str, Any]]:
        """Async: Perform optimizer step for this client's adapter.

        Args:
            optimizer_params: Optimizer configuration (e.g. {"name": "adam", "lr": 1e-4})
            immediate: If True, process the batch queue immediately instead of waiting for clock cycle.
        """
        return await self.create_async_future(
            request=self._optim_step_request(optimizer_params, immediate),
            endpoint="/optim_step",
        )

    def _save_checkpoint_request(self, checkpoint_path: str) -> SaveCheckpointRequest:
        return SaveCheckpointRequest(
            model_name=self.model_name,
            checkpoint_path=checkpoint_path,
            adapter_name=self.adapter_name,
        )

    def save_checkpoint(
        self, checkpoint_path: str
    ) -> TinkerbellFuture[SaveCheckpointResponse]:
        """Save model checkpoint (or specific adapter if lora_enabled)."""
        return self.create_future(
            request=self._save_checkpoint_request(checkpoint_path),
            endpoint="/save_checkpoint",
            parse_result_fn=lambda r: SaveCheckpointResponse(**r),
        )

    async def save_checkpoint_async(
        self, checkpoint_path: str
    ) -> AsyncTinkerbellFuture[SaveCheckpointResponse]:
        """Async: Save model checkpoint."""
        return await self.create_async_future(
            request=self._save_checkpoint_request(checkpoint_path),
            endpoint="/save_checkpoint",
            parse_result_fn=lambda r: SaveCheckpointResponse(**r),
        )

    def _push_to_hub_request(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        push_kwargs: Optional[dict[str, Any]] = None,
    ) -> PushToHubRequest:
        return PushToHubRequest(
            model_name=self.model_name,
            repo_id=repo_id,
            adapter_name=self.adapter_name,
            token=token,
            private=private,
            commit_message=commit_message,
            push_kwargs=push_kwargs or {},
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        push_kwargs: Optional[dict[str, Any]] = None,
    ) -> TinkerbellFuture[PushToHubResponse]:
        """Push model to Hugging Face Hub. Auto-creates repo if needed."""
        logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        return self.create_future(
            request=self._push_to_hub_request(
                repo_id, token, private, commit_message, push_kwargs
            ),
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
        """Async: Push model to Hugging Face Hub. Auto-creates repo if needed."""
        logger.info(f"Pushing model to Hugging Face Hub: {repo_id}")
        return await self.create_async_future(
            request=self._push_to_hub_request(
                repo_id, token, private, commit_message, push_kwargs
            ),
            endpoint="/push_to_hub",
            parse_result_fn=lambda r: PushToHubResponse(**r),
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
            # LoRA: Create actor with base model AND lora_paths set to initialize LoRA memory pool
            # SGLang requires lora_paths at startup for dynamic loading to work
            actor_base_model = self.base_model
            final_engine_kwargs["lora_paths"] = [checkpoint_path]
        else:
            # Full model: Create actor with base model, then load checkpoint weights
            # Disable LoRA mode since it's not needed
            actor_base_model = self.base_model
            final_engine_kwargs = {**final_engine_kwargs, "enable_lora": False}

        request = CreateSamplingActorRequest(
            base_model=actor_base_model,
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
            base_model=self.base_model,  # HF model path for tokenizer
            model_name=self.model_name,  # Actor name for routing
            adapter_name=self.adapter_name,
            timeout=self.timeout,
        )

        if wait_until_ready:
            sampling_client.wait_until_ready()

        # Always load the checkpoint - this handles:
        # - First run: loads the newly trained weights/adapter
        # - Re-run: reloads with updated weights after more training
        print(
            f"[save_weights_and_get_sampling_client] Loading checkpoint: {checkpoint_path}"
        )
        load_result = sampling_client.load_checkpoint(checkpoint_path).result()
        print(f"[save_weights_and_get_sampling_client] Load result: {load_result}")

        # Wait for checkpoint loading to complete (it's async on the server)
        sampling_client.wait_until_ready()
        print(
            f"[save_weights_and_get_sampling_client] Sampling client ready, lora_path={sampling_client.lora_path}"
        )

        return sampling_client
