from __future__ import annotations

import httpx

from tinkerbell.sampling.server import (
    DeployConfig,
    ModalSGLangServer,
    ServerConfig,
    SGLangServer,
)


class SamplingClient:
    def __init__(
        self,
        server_url: str,
        timeout: int = 600,
    ):
        self.server_url = server_url
        self.timeout = timeout

    def set_timeout(self, timeout: int):
        self.timeout = timeout

    def health(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.server_url}/health")
            return response.json()

    async def health_async(self) -> dict:
        with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.server_url}/health")
            return response.json()

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict,
        generate_kwargs: dict,
    ) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/batch_generate",
                json={
                    "prompts": prompts,
                    "sampling_params": sampling_params,
                    "generate_kwargs": generate_kwargs,
                },
            )
            return response.json()

    async def generate_async(
        self,
        prompts: list[str],
        sampling_params: dict,
        generate_kwargs: dict,
    ) -> dict:
        with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/batch_generate",
                json={
                    "prompts": prompts,
                    "sampling_params": sampling_params,
                    "generate_kwargs": generate_kwargs,
                },
            )
            return response.json()

    @classmethod
    def connect(
        cls,
        server_url: str,
        timeout: int = 600,
    ) -> SamplingClient:
        response = httpx.get(f"{server_url}/health")
        if response.status_code != 200:
            raise ValueError(f"Failed to connect to the server: {response.text}")
        return cls(
            server_url=server_url,
            timeout=timeout,
        )

    @classmethod
    def deploy(
        cls,
        server_config: ServerConfig,
        deploy_config: DeployConfig = DeployConfig(),
    ) -> SamplingClient:

        if deploy_config.deployment_type == "modal":
            server = ModalSGLangServer(
                config=server_config,
            )
        else:
            server = SGLangServer(
                config=server_config,
            )

        server_url = server.deploy(deploy_config)

        # Return a client connected to the deployed server
        return cls(
            server_url=server_url,
        )
