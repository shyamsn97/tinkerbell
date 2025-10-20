from __future__ import annotations

from typing import Optional

import httpx
from pydantic import BaseModel

from tinkerbell.sampling.server import DeployConfig, ModalSGLangServer, SGLangServer


class SamplingClientConfig(BaseModel):
    server_url: str
    server_port: Optional[int] = None
    timeout: int = 600


# class SamplingClientDeployConfig(BaseModel):
#     server_config: SGLangServerConfig
#     redeploy: bool = False


class SamplingClient:
    def __init__(
        self,
        server_url: str,
        server_port: Optional[int] = None,
        timeout: int = 600,
    ):
        self.server_url = server_url
        if server_port:
            self.server_url = f"{self.server_url}:{server_port}"
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
        server_port: Optional[int] = None,
        timeout: int = 600,
    ) -> SamplingClient:
        server_url = f"{server_url}"
        if server_port:
            server_url = f"{server_url}:{server_port}"
        response = httpx.get(f"{server_url}/health")
        if response.status_code != 200:
            raise ValueError(f"Failed to connect to the server: {response.text}")
        return cls(server_url=server_url, server_port=server_port, timeout=timeout)

    # @classmethod
    # def connect_or_deploy(
    #     cls,
    #     server_url: str,
    #     server_port: Optional[int] = None,
    #     timeout: int = 600,
    # ) -> SamplingClient:
    #     try:
    #         return cls.connect(server_url, server_port, timeout)
    #     except Exception as e:
    #         return cls.deploy(server_url, server_port, timeout)
    @classmethod
    def deploy(cls, *args, deploy_config: DeployConfig, **kwargs) -> SamplingClient:

        if deploy_config.deployment_type == "modal":
            server = ModalSGLangServer(*args, **kwargs)
        else:
            server = SGLangServer(*args, **kwargs)

        server.deploy(deploy_config)

        # Return a client connected to the deployed server
        return cls.connect(
            server_url=f"http://{deploy_config.host}",
            server_port=deploy_config.port,
        )
