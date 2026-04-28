"""TrainingClient: submits training ops and polls for results.

Every op returns a `JobHandle[T]`. Use `.result()` for sync,
`await` for async. Typed responses are parsed inline.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from tinker.types import Datum, LossFnType

from tinkerbell.client.common import BaseClient, JobHandle
from tinkerbell.types import (
    ActorStatusRequest,
    ActorStatusResponse,
    CreateSamplingActorRequest,
    CreateSamplingActorResponse,
    ForwardBackwardResponse,
    ForwardRequest,
    ForwardResponse,
    OptimStepRequest,
    PushToHubRequest,
    PushToHubResponse,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    ZeroGradRequest,
)
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


def _parse_forward(result: dict[str, Any]) -> ForwardResponse:
    return ForwardResponse(
        model_name=result.get("model_name", ""),
        logprobs=result.get("logprobs"),
        outputs=result.get("outputs"),
        metrics=result.get("metrics"),
    )


def _parse_forward_backward(result: dict[str, Any]) -> ForwardBackwardResponse:
    return ForwardBackwardResponse(
        model_name=result.get("model_name", ""),
        loss=result.get("loss"),
        logprobs=result.get("logprobs"),
        outputs=result.get("outputs"),
        metrics=result.get("metrics"),
        sum_gradient=result.get("sum_gradient"),
    )


class TrainingClient(BaseClient):
    """Client for training ops against the Tinkerbell API."""

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
        self.model_name = model_name or clean_model_name(base_model)
        self.adapter_name = adapter_name
        self.lora_enabled = lora_enabled
        self.lora_config = lora_config

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_actor_status(self) -> ActorStatusResponse:
        data = self.transport.submit(
            "/get_actor_status",
            ActorStatusRequest(model_name=self.model_name).model_dump(
                exclude_none=True
            ),
        )
        return ActorStatusResponse(**data)

    def wait_until_ready(
        self, poll_interval: float = 2.0, verbose: bool = True
    ) -> None:
        while True:
            status = self.get_actor_status()
            if verbose:
                logger.info(f"Actor status: {status.status}")
            if status.status == "ready":
                if verbose:
                    logger.info("Training actors are ready.")
                return
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Training ops
    # ------------------------------------------------------------------

    def zero_grad(self) -> JobHandle[dict[str, Any]]:
        req = ZeroGradRequest(
            model_name=self.model_name, adapter_name=self.adapter_name
        )
        return self._submit("/zero_grad", req.model_dump(exclude_none=True))

    def forward(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
    ) -> JobHandle[ForwardResponse]:
        req = ForwardRequest(
            model_name=self.model_name,
            data=data,
            forward_kwargs=forward_kwargs or {},
        )
        return self._submit(
            "/forward", req.model_dump(exclude_none=True), _parse_forward
        )

    def forward_backward(
        self,
        data: list[Datum],
        forward_kwargs: Optional[dict[str, Any]] = None,
        return_logprobs: bool = False,
        zero_grad: bool = True,
        optimizer_params: Optional[dict[str, Any]] = None,
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: Optional[dict[str, float]] = None,
    ) -> JobHandle[ForwardBackwardResponse]:
        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "data": [d.model_dump() for d in data],
            "forward_kwargs": forward_kwargs or {},
            "return_logprobs": return_logprobs,
            "zero_grad": zero_grad,
            "loss_fn": loss_fn,
        }
        if optimizer_params is not None:
            payload["optimizer_params"] = optimizer_params
        if loss_fn_config is not None:
            payload["loss_fn_config"] = loss_fn_config
        payload = {k: v for k, v in payload.items() if v is not None}
        return self._submit("/forward_backward", payload, _parse_forward_backward)

    def optim_step(
        self, optimizer_params: Optional[dict[str, Any]] = None
    ) -> JobHandle[dict[str, Any]]:
        req = OptimStepRequest(
            model_name=self.model_name,
            adapter_name=self.adapter_name,
            optimizer_params=optimizer_params or {},
        )
        return self._submit("/optim_step", req.model_dump(exclude_none=True))

    # ------------------------------------------------------------------
    # Side-effect ops
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, checkpoint_path: str
    ) -> JobHandle[SaveCheckpointResponse]:
        req = SaveCheckpointRequest(
            model_name=self.model_name,
            checkpoint_path=checkpoint_path,
            adapter_name=self.adapter_name,
        )
        return self._submit(
            "/save_checkpoint",
            req.model_dump(exclude_none=True),
            lambda r: SaveCheckpointResponse(**r),
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
        push_kwargs: Optional[dict[str, Any]] = None,
    ) -> JobHandle[PushToHubResponse]:
        logger.info(f"Submitting push_to_hub: {repo_id}")
        req = PushToHubRequest(
            model_name=self.model_name,
            repo_id=repo_id,
            adapter_name=self.adapter_name,
            token=token,
            private=private,
            commit_message=commit_message,
            push_kwargs=push_kwargs or {},
        )
        return self._submit(
            "/push_to_hub",
            req.model_dump(exclude_none=True),
            lambda r: PushToHubResponse(**r),
        )

    # ------------------------------------------------------------------
    # Save-and-sample helper
    # ------------------------------------------------------------------

    def save_weights_and_get_sampling_client(
        self,
        checkpoint_path: str,
        tp_size: Optional[int] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = False,
    ):
        from tinkerbell.client.sampling import SamplingClient

        self.save_checkpoint(checkpoint_path).result()

        is_lora = self.adapter_name is not None
        final_kwargs = dict(engine_kwargs or {})
        if is_lora:
            final_kwargs["lora_paths"] = [checkpoint_path]
        else:
            final_kwargs["enable_lora"] = False

        create_req = CreateSamplingActorRequest(
            base_model=self.base_model,
            model_name=self.model_name,
            tp_size=tp_size or 1,
            engine_kwargs=final_kwargs,
        )
        resp = CreateSamplingActorResponse(
            **self.transport.submit("/create_sampling_actor", create_req.model_dump())
        )
        if not resp.success:
            raise RuntimeError(f"Failed to create sampling actor: {resp.message}")

        sampling_client = SamplingClient(
            server_url=self.server_url,
            base_model=self.base_model,
            model_name=self.model_name,
            adapter_name=self.adapter_name,
            timeout=self.timeout,
        )
        if wait_until_ready:
            sampling_client.wait_until_ready()
        sampling_client.load_checkpoint(checkpoint_path).result()
        sampling_client.wait_until_ready()
        return sampling_client
