from __future__ import annotations

import logging
import time
from typing import Any, Optional

from tinkerbell.client.base import BaseClient
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.client.training import TrainingClient
from tinkerbell.types import (
    CreateSamplingActorRequest,
    CreateTrainingActorsRequest,
    DeployConfig,
    HealthResponse,
)
from tinkerbell.types.responses import CreateSamplingActorResponse
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


class ServiceClient(BaseClient):
    def __init__(self, server_url: str | None = None, timeout: float = 600.0):
        super().__init__(server_url, timeout)

    def get_health(
        self, max_retries: int = 5, retry_delay: float = 2.0
    ) -> HealthResponse:
        """Get health with exponential backoff retry."""
        import httpx

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
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Health check failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Health check failed after {max_retries} attempts")
                    raise

    def is_deployed(self) -> bool:
        try:
            _ = self.get_health()
            return True
        except Exception:
            return False

    def get_ray_actors(self) -> list[str]:
        """Get list of all Ray actors from the server."""
        response = self.client.get("/get_ray_actors")
        response.raise_for_status()
        result = response.json()
        return result.get("actor_names", [])

    def get_store_keys(self) -> list[str]:
        """Get list of all keys from the global store."""
        response = self.client.get("/get_store_keys")
        response.raise_for_status()
        result = response.json()
        return result.get("keys", [])

    def wait_until_ready(self, max_retries: int = 30, retry_delay: float = 2.0) -> bool:
        """Wait until server is ready by polling health endpoint."""
        logger.info("Waiting for server to be ready...")
        for attempt in range(max_retries):
            try:
                self.get_health(max_retries=1, retry_delay=0.1)
                logger.info("Server is ready!")
                return True
            except Exception:
                if attempt < max_retries - 1:
                    logger.info(
                        f"Server not ready (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
        logger.error(f"Server did not become ready after {max_retries} attempts")
        return False

    @classmethod
    def deploy(
        cls,
        deploy_config: DeployConfig,
        wait_for_ready: bool = True,
        timeout: float = 600.0,
    ) -> ServiceClient:
        """Deploy the server."""
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
    ) -> ServiceClient:
        """Deploy if not deployed, otherwise connect to existing server."""
        server_url = deploy_config.server_url
        if not redeploy:
            try:
                server_url = deploy_config.connect()
                logger.info(f"Connected to existing server at: {server_url}")
            except Exception:
                redeploy = True
        if redeploy:
            server_url = deploy_config.deploy()
            print(f"Deployed new server at: {server_url}")
        if server_url is None:
            raise ValueError("Server URL is None. Please check the deploy config.")
        service_client = cls(server_url=server_url, timeout=timeout)
        if wait_for_ready:
            service_client.wait_until_ready()
        return service_client

    def _check_deployed(self):
        if not self.is_deployed():
            raise ValueError("Server is not deployed. Please deploy the server first.")

    def create_training_client(
        self,
        base_model: str,
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
        """Create training actors. Args: base_model, tp_size, model_name, adapter_name, lora_config, etc."""
        self._check_deployed()
        model_name = model_name or clean_model_name(base_model)

        request = CreateTrainingActorsRequest(
            base_model=base_model,
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
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
            timeout=self.timeout,
            lora_enabled=lora_config is not None,
            lora_config=lora_config_dict,
        )

    def create_sampling_client(
        self,
        base_model: str,
        tp_size: int = 1,
        model_name: Optional[str] = None,
        adapter_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        wait_until_ready: bool = False,
    ) -> SamplingClient:
        """Create sampling actors. Args: base_model, tp_size, model_name, adapter_name, checkpoint_path, etc."""
        self._check_deployed()
        model_name = model_name or clean_model_name(base_model)
        is_lora = adapter_name is not None
        final_engine_kwargs = engine_kwargs or {}

        if is_lora:
            actor_base_model = base_model
        else:
            actor_base_model = checkpoint_path or base_model
            final_engine_kwargs = {**final_engine_kwargs, "enable_lora": False}

        request = CreateSamplingActorRequest(
            base_model=actor_base_model,
            model_name=model_name,
            tp_size=tp_size,
            engine_kwargs=final_engine_kwargs,
        )
        response = self.client.post("/create_sampling_actor", json=request.model_dump())
        response.raise_for_status()
        result = CreateSamplingActorResponse(**response.json())
        if not result.success:
            raise RuntimeError(f"Failed to create sampling actor: {result.message}")

        sampling_client = SamplingClient(
            server_url=self.server_url,
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
            timeout=self.timeout,
        )
        if wait_until_ready:
            sampling_client.wait_until_ready()
        if is_lora and checkpoint_path:
            sampling_client.load_checkpoint(checkpoint_path).result()
        return sampling_client
