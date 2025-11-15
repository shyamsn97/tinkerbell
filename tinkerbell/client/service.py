from typing import Any, Optional

import httpx

from tinkerbell.client.training import TrainingClient
from tinkerbell.types import CreateTrainingActorsRequest, DeployConfig, HealthResponse


class ServiceClient:
    def __init__(self, server_url: str | None = None, timeout: float = 600.0):
        self.server_url = server_url
        self.timeout = timeout

    def get_health(self) -> HealthResponse:
        with httpx.Client(base_url=self.server_url, timeout=self.timeout) as client:
            response = client.get("/health")
            response.raise_for_status()
            return HealthResponse(**response.json())

    def is_deployed(self) -> bool:
        try:
            _ = self.get_health()
            return True
        except Exception:
            return False

    def deploy(self, deploy_config: DeployConfig) -> str:
        if self.server_url is None:
            self.server_url = deploy_config.deploy()
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
        )

        with httpx.Client(base_url=self.server_url, timeout=self.timeout) as client:
            response = client.post(
                "/create_training_actors",
                json=request.model_dump(),
            )
            response.raise_for_status()
            return TrainingClient(
                server_url=self.server_url,
                model_id=model_id,
                timeout=self.timeout,
            )
