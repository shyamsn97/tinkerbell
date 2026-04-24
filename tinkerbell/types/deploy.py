import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class DeployConfig(BaseModel):
    server_url: str = "https://0.0.0.0:8000"
    max_wait_time: float = 600.0
    clock_cycle: float = 2.0

    @property
    def deployment_type(self) -> str:
        return "local"

    def connect(self) -> str:
        raise NotImplementedError("Connect is not implemented for local deployment")

    def deploy(self) -> str:
        raise NotImplementedError("Deploy is not implemented for local deployment")


class ModalDeployConfig(DeployConfig):
    gpu: str = "H100"
    num_gpus: int = 1
    timeout: int = 86400
    scaledown_window: int = 600
    max_inputs: int = 100

    @property
    def deployment_type(self) -> str:
        return "modal"

    def connect(self) -> str:
        try:
            import httpx
            import modal

            existing_function = modal.Function.from_name("tinkerbell-service", "serve")
            existing_url = existing_function.web_url

            # Verify the server is actually active by checking health endpoint
            try:
                response = httpx.get(f"{existing_url}/health", timeout=5.0)
                response.raise_for_status()
                logger.info(f"Server is active and responding at: {existing_url}")
                return existing_url
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.HTTPStatusError,
            ) as e:
                logger.warning(
                    f"Found Modal function but server is not responding: {e}. "
                    "Treating as if no server exists."
                )
                raise ConnectionError(
                    f"Modal function exists but server is not responding: {e}"
                ) from e
        except Exception as e:
            raise e

    def deploy(self) -> str:
        from tinkerbell.api.deploy import deploy_on_modal

        modal_url = deploy_on_modal(
            server_url=self.server_url,
            max_wait_time=self.max_wait_time,
            clock_cycle=self.clock_cycle,
            gpu=self.gpu,
            num_gpus=self.num_gpus,
            timeout=self.timeout,
            scaledown_window=self.scaledown_window,
            max_inputs=self.max_inputs,
        )
        return modal_url
