from typing import Any, Dict

import ray
from ray import serve

from tinkerbell.models import (  # ForwardBackwardResponse,
    ActorStatusRequest,
    ActorStatusResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ForwardBackwardRequest,
    HealthResponse,
    ModalDeployConfig,
    RemoteFuture,
)
from tinkerbell.service.base import APP
from tinkerbell.training.manager import TrainingManager
from tinkerbell.utils import get_host_and_port


class TinkerbellServiceDeployment:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.training_manager = TrainingManager()

    @APP.get("/health")
    async def health(self) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            name="TinkerbellService",
        )

    @APP.post("/create_training_actors")
    async def create_training_actors(
        self,
        request: CreateTrainingActorsRequest,
    ) -> CreateTrainingActorsResponse:
        model_name = await self.training_manager.create_training_actors(
            request.world_size,
            request.master_addr,
            request.master_port,
            request.model_name,
            request.model_kwargs,
            request.parallelize_plan,
            request.optimizer_params,
            request.scheduler_params,
            request.ray_worker_options,
        )
        return CreateTrainingActorsResponse(
            success=True,
            model_name=model_name,
            message=f"Training actors for model {model_name} created...",
        )

    @APP.post("/forward_backward")
    async def forward_backward(
        self,
        request: ForwardBackwardRequest,
    ) -> RemoteFuture:
        if not self.training_manager.running:
            await self.training_manager.start()
        return await self.training_manager.forward_backward(
            request.model_name, request.inputs, request.forward_kwargs
        )

    @APP.post("/get_actor_status")
    async def get_actor_status(
        self,
        request: ActorStatusRequest,
    ) -> ActorStatusResponse:
        status = await self.training_manager.get_actor_status(request.model_name)
        return ActorStatusResponse(
            status=status.value,
            message=f"Actor status for model {request.model_name} is {status.value}",
        )

    @APP.post("/get_result")
    async def get_result(
        self,
        request: RemoteFuture,
    ) -> Dict[str, Any]:
        return await self.training_manager.get_result(request.request_id)


def deploy_service(server_url: str, **deployment_kwargs):
    """Deploy the TinkerbellService with Ray Serve.

    Args:
        server_url: URL of the backend training server
        host: Host to bind the service to
        port: Port to bind the service to
        **deployment_kwargs: Optional Ray Serve deployment configurations
            (e.g., num_replicas, ray_actor_options, autoscaling_config)
    """
    if not ray.is_initialized():
        ray.init()

    host, port = get_host_and_port(server_url)
    serve.start(detached=True, http_options={"host": host, "port": port})

    # Apply any custom deployment options if provided

    deployment_kwargs["ray_actor_options"] = {"num_gpus": 0}

    deployment = serve.deployment(**deployment_kwargs)(
        serve.ingress(APP)(TinkerbellServiceDeployment)
    )

    serve.run(
        deployment.bind(server_url),
    )

    return server_url


def deploy_on_modal(
    server_url: str,
    deploy_config: ModalDeployConfig = ModalDeployConfig(),
):
    """Deploy the TinkerbellService on Modal.

    Args:
        server_url: URL of the backend training server
        **deployment_kwargs: Optional Modal deployment configurations
            (e.g., num_replicas, ray_actor_options, autoscaling_config)
    """
    try:
        import modal
        from modal import runner
    except ImportError:
        raise ImportError("Modal is not installed. Install it with: pip install modal")

    import os

    app = modal.App(name="tinkerbell-service")

    env_variables = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", None),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "NCCL_DEBUG": "INFO",  # For debugging NCCL issues
        "TORCH_DISTRIBUTED_BACKEND": "nccl",  # Prefer NCCL
        # Add these for better visibility:
        "TORCH_DISTRIBUTED_DEBUG": "INFO",  # Shows distributed init details
        "TORCH_SHOW_CPP_STACKTRACES": "1",  # If it crashes
        "SGLANG_LOG_LEVEL": "DEBUG",  # If SGLang respects this
        "TRANSFORMERS_VERBOSITY": "info",  # See model loading progress
        "RAY_DEDUP_LOGS": "0",
    }

    # Define Modal image with required dependencies
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12"
        )
        .apt_install("libnuma-dev", "build-essential", "clang")
        .env({"CUDA_HOME": "/usr/local/cuda"})  # Add this line
        .pip_install(
            "torch==2.4.0",
            extra_index_url="https://download.pytorch.org/whl/cu126",
        )
        .uv_pip_install(
            "pybase64",
            "zmq",
            "xformers",
            "transformers",
            "numpy",
            "fastapi",
            "uvicorn",
            "pydantic",
            "cloudpickle",
            "dill",
            "flashinfer-python",  # Install FlashInfer first
            "sglang[all]==0.5.2",
            "sgl-kernel",
            "huggingface_hub",
            "hf_transfer",
            "ray",
            "ray[serve]",
        )
        .env(env_variables)
    )
    image = image.add_local_python_source("tinkerbell")

    # Create a volume for model checkpoints if needed
    volume = modal.Volume.from_name("tinkerbell-checkpoints", create_if_missing=True)

    @app.function(
        image=image,
        gpu=f"{deploy_config.gpu}:{deploy_config.num_gpus}",
        volumes={"/checkpoints": volume},
        timeout=deploy_config.timeout,
        container_idle_timeout=deploy_config.container_idle_timeout,
        serialized=True,
    )
    @modal.concurrent(max_inputs=deploy_config.max_inputs)
    @modal.web_server(
        8000,
        label="training-service",
    )
    def serve():
        # Use Ray Serve within Modal
        deploy_service(server_url)

    with modal.enable_output():
        # Deploy the app
        runner.deploy_app(app)
