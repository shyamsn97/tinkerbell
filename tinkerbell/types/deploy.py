from ._models import BaseModel


class DeployConfig(BaseModel):
    server_url: str = "https://0.0.0.0:8000"
    max_wait_time: float = 600.0
    clock_cycle: float = 10.0

    @property
    def deployment_type(self) -> str:
        return "local"

    def deploy(self) -> str:
        return self.server_url


class ModalDeployConfig(DeployConfig):
    gpu: str = "H100"
    num_gpus: int = 1
    timeout: int = 86400
    container_idle_timeout: int = 600
    max_inputs: int = 100

    @property
    def deployment_type(self) -> str:
        return "modal"

    def deploy(self) -> str:
        from tinkerbell.service.server import deploy_on_modal

        modal_url = deploy_on_modal(
            server_url=self.server_url,
            max_wait_time=self.max_wait_time,
            clock_cycle=self.clock_cycle,
            gpu=self.gpu,
            num_gpus=self.num_gpus,
            timeout=self.timeout,
            container_idle_timeout=self.container_idle_timeout,
            max_inputs=self.max_inputs,
        )
        return modal_url
