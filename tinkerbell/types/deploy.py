from pydantic import BaseModel


class DeployConfig(BaseModel):
    @property
    def deployment_type(self) -> str:
        return "local"


class ModalDeployConfig(DeployConfig):
    gpu: str = "H100"
    num_gpus: int = 1
    timeout: int = 86400
    serialized: bool = True
    container_idle_timeout: int = 600
    max_inputs: int = 100

    @property
    def deployment_type(self) -> str:
        return "modal"
