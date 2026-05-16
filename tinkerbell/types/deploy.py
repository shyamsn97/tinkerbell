"""Minimal server deployment config."""

from __future__ import annotations

from tinkerbell.types.base import BaseModel


class DeployConfig(BaseModel):
    server_url: str = "http://127.0.0.1:8000"
    namespace: str = "tinkerbell"
    max_wait_time: float = 600.0
    clock_cycle: float = 2.0

    def connect(self) -> str:
        raise ConnectionError("No existing local server connection is configured")

    def deploy(self) -> str:
        from tinkerbell.api.server import deploy_service

        return deploy_service(server_url=self.server_url, namespace=self.namespace)


class ModalDeployConfig(DeployConfig):
    gpu: str = "H100"
    num_gpus: int = 1
    timeout: int = 86400
    scaledown_window: int = 600
    max_inputs: int = 100
    max_containers: int = 1
    memory_mb: int = 8192

    def connect(self) -> str:
        import httpx
        import modal

        existing_function = modal.Function.from_name("tinkerbell-service", "serve")
        existing_url = existing_function.web_url
        response = httpx.get(f"{existing_url}/health", timeout=5.0)
        response.raise_for_status()
        health = response.json()
        if (
            health.get("name") != "TinkerbellServer"
            or health.get("job_protocol") != "ray_internal_kv"
            or health.get("container_model") != "single"
        ):
            raise ConnectionError("existing Modal function is not the current server")
        return existing_url

    def deploy(self) -> str:
        from tinkerbell.api.deploy import deploy_on_modal

        return deploy_on_modal(
            server_url=self.server_url,
            gpu=self.gpu,
            num_gpus=self.num_gpus,
            timeout=self.timeout,
            scaledown_window=self.scaledown_window,
            max_inputs=self.max_inputs,
            max_containers=self.max_containers,
            memory_mb=self.memory_mb,
        )


__all__ = ["DeployConfig", "ModalDeployConfig"]
