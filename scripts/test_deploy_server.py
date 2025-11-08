from tinkerbell.service.server import deploy_on_modal
from tinkerbell.models import ModalDeployConfig

if __name__ == "__main__":
    deploy_config = ModalDeployConfig(
        gpu="A100",
        num_gpus=5,
        timeout=86400,
        container_idle_timeout=600,
        max_inputs=200,
    )
    server_url = "https://0.0.0.0:8000"
    deploy_on_modal(
        server_url=server_url,
        max_wait_time=300.0,
        clock_cycle=10.0,
        deploy_config=deploy_config,
    )
