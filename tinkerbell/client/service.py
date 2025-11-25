import logging
import time
from typing import Any, Optional

from tinkerbell.client.base import BaseClient, TinkerbellFuture
from tinkerbell.client.training import TrainingClient
from tinkerbell.types import (
    CreateTrainingActorsRequest,
    DeployConfig,
    GetRayActorsResponse,
    HealthResponse,
)
from tinkerbell.types.responses import RemoteFuture

logger = logging.getLogger(__name__)


class ServiceClient(BaseClient):
    def __init__(self, server_url: str | None = None, timeout: float = 600.0):
        super().__init__(server_url, timeout)

    def get_health(
        self, max_retries: int = 5, retry_delay: float = 2.0
    ) -> HealthResponse:
        """Get health synchronously with retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (doubles each retry)
        """
        import httpx

        last_exception = None
        delay = retry_delay

        for attempt in range(max_retries):
            try:
                response = self.client.get("/health")
                response.raise_for_status()
                return HealthResponse(**response.json())
            except (
                httpx.RemoteProtocolError,
                httpx.ConnectError,
                httpx.TimeoutException,
            ) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Health check failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Health check failed after {max_retries} attempts")
                    raise

        # Should not reach here, but just in case
        raise last_exception if last_exception else Exception("Health check failed")

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
        remote_future_dict = response.json()

        def _parse_result(result: dict[str, Any]):
            return result["actor_names"]

        return self.create_future_from_request_id(
            remote_future=RemoteFuture(**remote_future_dict),
            parse_result_fn=_parse_result,
        )

    def get_store_keys(self) -> list[str]:
        """Get list of all keys from the global store."""
        response = self.client.get("/get_store_keys")
        response.raise_for_status()
        result = response.json()
        return result.get("keys", [])

    def wait_until_ready(self, max_retries: int = 30, retry_delay: float = 2.0) -> bool:
        """Wait until server is ready by polling health endpoint.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if server becomes ready, False otherwise
        """
        logger.info("Waiting for server to be ready...")
        for attempt in range(max_retries):
            try:
                self.get_health(max_retries=1, retry_delay=0.1)
                logger.info("Server is ready!")
                return True
            except Exception:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Server not ready yet (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Server did not become ready after {max_retries} attempts"
                    )
                    return False
        return False

    def deploy(
        self,
        deploy_config: DeployConfig,
        redeploy: bool = False,
        wait_for_ready: bool = True,
    ) -> str:
        """Deploy the server.

        Args:
            deploy_config: Deployment configuration
            redeploy: If True, redeploy even if server_url is already set
            wait_for_ready: If True, wait for server to be ready before returning

        Returns:
            Server URL
        """
        if self.server_url is None or redeploy:
            self.server_url = deploy_config.deploy()

        if wait_for_ready:
            self.wait_until_ready()

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
