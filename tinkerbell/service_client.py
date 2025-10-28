import logging
from typing import Any

from pydantic import BaseModel

from tinkerbell.sampling.client import SamplingClient
from tinkerbell.sampling.server import DeployConfig, ServerConfig

logger = logging.getLogger(__name__)


class CreateTrainingClientConfig(BaseModel):
    model_name: str | None = None
    model_config: dict | None = None
    lora_config: dict | None = None
    model_checkpoint_path: str | None = None
    deploy_config: dict | None = None


class ServiceClient:
    def __init__(self):
        self.clients = {}

    # def create_training_client(
    #     self,
    #     config: CreateTrainingClientConfig | None = None,
    #     redeploy: bool = False,
    #     **kwargs,
    # ) -> Any:
    #     """Create a client for the service.

    #     Args:
    #         **kwargs: Keyword arguments to create the client.

    #     Returns:
    #         Any: A client for the service.
    #     """
    #     if config is None:
    #         config = CreateTrainingClientConfig()
    #     config_dict = config.model_dump()
    #     merged_config_dict = {**config_dict, **kwargs}
    #     config = CreateTrainingClientConfig(**merged_config_dict)

    #     self.clients[config.model_name] = create_training_client(
    #         config, redeploy=redeploy
    #     )

    def create_sampling_client(
        self,
        server_config: ServerConfig,
        deploy_config: DeployConfig = DeployConfig(),
        redeploy: bool = False,
    ) -> Any:
        """Create a client for the service.

        Args:
            **kwargs: Keyword arguments to create the client.

        Returns:
            Any: A client for the service.
        """
        # if config is None:
        #     config = CreateSamplingClientConfig()
        # config_dict = config.model_dump()
        # merged_config_dict = {**config_dict, **kwargs}
        # config = CreateSamplingClientConfig(**merged_config_dict)

        server_url = server_config.server_url
        client = None
        if not redeploy:
            try:
                client = SamplingClient.connect(server_url)
                logger.info(
                    f"Connected to existing client for {server_config.model_name}"
                )
            except Exception:
                client = None
        if client is None:
            client = SamplingClient.deploy(
                server_config=server_config,
                deploy_config=deploy_config,
            )
        self.clients[server_config.model_name] = client
        return client
