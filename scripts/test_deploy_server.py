from tinkerbell.service.server import deploy_on_modal
from tinkerbell.models import ModalDeployConfig

if __name__ == "__main__":
    deploy_config = ModalDeployConfig(
        gpu="H100",
        num_gpus=4,
        timeout=86400,
        container_idle_timeout=600,
        max_inputs=100,
    )
    server_url = "https://0.0.0.0:8000"
    deploy_on_modal(server_url, deploy_config)