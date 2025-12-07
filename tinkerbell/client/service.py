import logging
import time
from typing import Any, Optional

from tinkerbell.client.base import BaseClient, TinkerbellFuture
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.client.training import TrainingClient
from tinkerbell.types import (
    CreateSamplingActorRequest,
    CreateTrainingActorsRequest,
    DeployConfig,
    GetRayActorsResponse,
    HealthResponse,
)
from tinkerbell.types.responses import RemoteFuture

logger = logging.getLogger(__name__)

# # Configure logging to ensure messages show up in terminal
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     force=True,  # Override any existing configuration
# )


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

    @classmethod
    def deploy(
        cls,
        deploy_config: DeployConfig,
        wait_for_ready: bool = True,
        timeout: float = 600.0,
    ) -> str:
        """Deploy the server.

        Args:
            deploy_config: Deployment configuration
            redeploy: If True, redeploy even if server_url is already set
            wait_for_ready: If True, wait for server to be ready before returning

        Returns:
            Server URL
        """
        return cls.deploy_or_connect(
            deploy_config, wait_for_ready=wait_for_ready, redeploy=True, timeout=timeout
        )

    @classmethod
    def deploy_or_connect(
        cls,
        deploy_config: DeployConfig,
        redeploy: bool = False,
        wait_for_ready: bool = True,
        timeout: float = 600.0,
    ) -> str:
        """Deploy the server if not deployed, otherwise connect to the existing server."""
        deploy = False
        server_url = deploy_config.server_url
        try:
            logger.info("Trying to connect to existing server...")
            if not redeploy:
                server_url = deploy_config.connect()
                logger.info(f"Connected to existing server at: {server_url}")
        except Exception:
            logger.info("No existing server found, deploying new one...")
            deploy = True
        if deploy or redeploy:
            server_url = deploy_config.deploy()
            print(f"Deployed new server at: {server_url}")
        if server_url is None:
            raise ValueError("Server URL is None. Please check the deploy config.")
        service_client = cls(server_url=server_url, timeout=timeout)

        if wait_for_ready:
            service_client.wait_until_ready()

        return server_url

    def create_training_client(
        self,
        model_id: str,
        tp_size: int,
        model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        parallelize_plan: Optional[dict[str, str]] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        scheduler_params: Optional[dict[str, Any]] = None,
        lora_config: Optional[dict[str, Any]] = None,
        ray_worker_options: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = False,
        initialize_random_weights: bool = False,
    ) -> TrainingClient:
        """Create training actors on the server.

        Args:
            model_id: HuggingFace model ID
            tp_size: Tensor parallel size (world_size)
            model_name: Group name for actor sharing. Same model_name = shared actors.
            adapter_name: Name for this LoRA adapter (for multi-LoRA)
            lora_config: LoRA configuration (None for full model training)
        """
        if not self.is_deployed():
            raise ValueError("Server is not deployed. Please deploy the server first.")

        model_name = model_name or model_id

        request = CreateTrainingActorsRequest(
            model_id=model_id,
            model_name=model_name,
            adapter_name=adapter_name,
            world_size=tp_size,
            parallelize_plan=parallelize_plan or {},
            model_kwargs=model_kwargs or {},
            scheduler_params=scheduler_params or {},
            lora_config=lora_config,
            ray_worker_options=ray_worker_options or {},
            wait_until_ready=wait_until_ready,
            initialize_random_weights=initialize_random_weights,
        )

        response = self.client.post(
            "/create_training_actors", json=request.model_dump()
        )
        response.raise_for_status()

        lora_config_dict = (
            lora_config.model_dump()
            if hasattr(lora_config, "model_dump")
            else lora_config
        )

        return TrainingClient(
            server_url=self.server_url,
            model_id=model_id,
            model_name=model_name,
            adapter_name=adapter_name,
            timeout=self.timeout,
            lora_enabled=lora_config is not None,
            lora_config=lora_config_dict,
        )

    def create_sampling_client(
        self,
        model_id: str,
        tp_size: int,
        model_name: Optional[str] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
    ) -> SamplingClient:
        """Create sampling actors on the server."""
        if not self.is_deployed():
            raise ValueError("Server is not deployed. Please deploy the server first.")

        request = CreateSamplingActorRequest(
            model_id=model_id,
            model_name=model_name,
            tp_size=tp_size,
            engine_kwargs=engine_kwargs or {},
        )

        response = self.client.post("/create_sampling_actor", json=request.model_dump())
        response.raise_for_status()
