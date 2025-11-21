import logging
import time
from typing import Any, Optional

import httpx

from tinkerbell.client.base import BaseClient, TinkerbellFuture
from tinkerbell.client.training import TrainingClient
from tinkerbell.types import (
    CreateTrainingActorsRequest,
    DeployConfig,
    GetRayActorsResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


class ServiceClient(BaseClient):
    def __init__(self, server_url: str | None = None, timeout: float = 600.0):
        super().__init__(server_url, timeout)

    def get_health(self) -> HealthResponse:
        """Get health synchronously (no async on server for this endpoint)."""
        response = self.client.get("/health")
        response.raise_for_status()
        return HealthResponse(**response.json())

    def is_deployed(self) -> bool:
        try:
            _ = self.get_health()
            return True
        except Exception:
            return False

    def get_ray_actors(self) -> TinkerbellFuture[GetRayActorsResponse]:
        """Get list of all Ray actors from the server."""
        # Send request immediately
        response = self.client.post("/get_ray_actors", json={})
        response.raise_for_status()
        remote_future = response.json()
        
        # Parse result
        def _parse_result(result: dict[str, Any]) -> GetRayActorsResponse:
            return GetRayActorsResponse(**result)
        
        return TinkerbellFuture(
            request_id=remote_future["request_id"],
            server_url=self.server_url,
            poll_endpoint="/poll_result",
            result_parser=_parse_result,
            poll_interval=0.1,
            timeout=self.timeout,
        )

    def deploy(self, deploy_config: DeployConfig) -> str:
        if self.server_url is None:
            self.server_url = deploy_config.deploy()

        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        while not self.is_deployed():
            time.sleep(1)
        logger.info("Server is ready!")
        return self.server_url

    def create_training_client(
        self,
        model_id: str,
        tp_size: int,
        parallelize_plan: Optional[dict[str, str]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        scheduler_params: Optional[dict[str, Any]] = None,
        lora_config: Optional[dict[str, Any]] = None,
        ray_worker_options: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = False,
        deploy_config: DeployConfig | None = None,
        initialize_random_weights: bool = False,
    ) -> TrainingClient:
        """
        Create training actors on the server.

        Args:
            model_id: ID of the model
            tp_size: Number of processes in distributed training (world_size)
            parallelize_plan: Dictionary mapping layer patterns to parallelization strategy
            model_kwargs: Additional kwargs for model initialization
            scheduler_params: Learning rate scheduler parameters
            lora_config: LoRA configuration
            ray_worker_options: Ray worker options
            wait_until_ready: If True, blocks until actors are ready
            deploy_config: Deployment configuration

        Returns:
            TrainingClient
        """
        if not self.is_deployed() and deploy_config is not None:
            self.deploy(deploy_config)

        request = CreateTrainingActorsRequest(
            model_id=model_id,
            world_size=tp_size,
            parallelize_plan=parallelize_plan or {},
            model_kwargs=model_kwargs or {},
            scheduler_params=scheduler_params or {},
            lora_config=lora_config or {},
            ray_worker_options=ray_worker_options or {},
            wait_until_ready=wait_until_ready,
            initialize_random_weights=initialize_random_weights,
        )

        response = self.client.post(
            "/create_training_actors",
            json=request.model_dump(),
        )
        response.raise_for_status()

        # Convert LoraConfig to dict if needed
        lora_config_dict = None
        if lora_config:
            if hasattr(lora_config, "model_dump"):
                lora_config_dict = lora_config.model_dump()
            else:
                lora_config_dict = lora_config

        return TrainingClient(
            server_url=self.server_url,
            model_id=model_id,
            timeout=self.timeout,
            lora_enabled=True,
            lora_config=lora_config_dict,
        )
