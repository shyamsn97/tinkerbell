"""ServiceClient: deploy/connect + factory for TrainingClient / SamplingClient."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from tinkerbell.client.common import BaseClient
from tinkerbell.client.sampling import SamplingClient
from tinkerbell.client.training import TrainingClient
from tinkerbell.types import (
    CreateSamplingActorRequest,
    CreateSamplingActorResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    DeployConfig,
    HealthResponse,
)
from tinkerbell.utils import clean_model_name

logger = logging.getLogger(__name__)


class ServiceClient(BaseClient):
    def __init__(self, server_url: str | None = None, timeout: float = 600.0):
        super().__init__(server_url=server_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Health / lifecycle
    # ------------------------------------------------------------------

    def get_health(self) -> HealthResponse:
        """One-shot health probe. Raises on any transport/HTTP error."""
        r = self.transport.sync.get("/health")
        r.raise_for_status()
        return HealthResponse(**r.json())

    def is_deployed(self) -> bool:
        try:
            self.get_health()
            return True
        except Exception:
            return False

    def wait_until_ready(
        self, timeout: float = 60.0, poll_interval: float = 2.0
    ) -> bool:
        """Block until /health returns 200 or timeout elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.get_health()
                logger.info("Gateway is ready.")
                return True
            except Exception as e:
                logger.info(f"Gateway not ready ({e}); retrying in {poll_interval}s")
                time.sleep(poll_interval)
        logger.error(f"Gateway did not become ready within {timeout}s")
        return False

    # ------------------------------------------------------------------
    # Deploy
    # ------------------------------------------------------------------

    @classmethod
    def deploy(
        cls,
        deploy_config: DeployConfig,
        wait_for_ready: bool = True,
        timeout: float = 600.0,
    ) -> "ServiceClient":
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
    ) -> "ServiceClient":
        server_url = deploy_config.server_url
        if not redeploy:
            try:
                server_url = deploy_config.connect()
                logger.info(f"Connected to existing gateway at: {server_url}")
            except Exception:
                redeploy = True
        if redeploy:
            server_url = deploy_config.deploy()
            logger.info(f"Deployed new gateway at: {server_url}")
        if server_url is None:
            raise ValueError("Server URL is None. Check the deploy config.")
        client = cls(server_url=server_url, timeout=timeout)
        if wait_for_ready:
            client.wait_until_ready()
        return client

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def _check_deployed(self):
        if not self.is_deployed():
            raise ValueError("Gateway is not deployed.")

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
        initialize_base_model: bool = False,
    ) -> TrainingClient:
        self._check_deployed()
        model_name = model_name or clean_model_name(base_model)

        req = CreateTrainingActorsRequest(
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
            initialize_base_model=initialize_base_model,
        )
        resp = CreateTrainingActorsResponse(
            **self.transport.submit("/create_training_actors", req.model_dump())
        )
        if not resp.success:
            raise RuntimeError(f"create_training_actors failed: {resp.message}")

        lora_dict = (
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
            lora_config=lora_dict,
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
        self._check_deployed()
        model_name = model_name or clean_model_name(base_model)
        is_lora = adapter_name is not None
        final_kwargs = dict(engine_kwargs or {})
        if is_lora:
            actor_base_model = base_model
        else:
            actor_base_model = checkpoint_path or base_model
            final_kwargs["enable_lora"] = False

        create_req = CreateSamplingActorRequest(
            base_model=actor_base_model,
            model_name=model_name,
            tp_size=tp_size,
            engine_kwargs=final_kwargs,
        )
        resp = CreateSamplingActorResponse(
            **self.transport.submit("/create_sampling_actor", create_req.model_dump())
        )
        if not resp.success:
            raise RuntimeError(f"create_sampling_client failed: {resp.message}")

        client = SamplingClient(
            server_url=self.server_url,
            base_model=base_model,
            model_name=model_name,
            adapter_name=adapter_name,
            timeout=self.timeout,
        )
        if wait_until_ready:
            client.wait_until_ready()
        if is_lora and checkpoint_path:
            client.load_checkpoint(checkpoint_path).result()
        return client
