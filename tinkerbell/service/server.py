from typing import Any, Dict

import ray
from fastapi import FastAPI
from ray import serve

from tinkerbell.training.manager import TrainingManager
from tinkerbell.types import (
    ActorStatusRequest,
    ActorStatusResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GetRayActorsResponse,
    HealthResponse,
    ModalDeployConfig,
    RemoteFuture,
)
from tinkerbell.types.optimizer import (
    OptimStepRequest,
    OptimStepResponse,
    ZeroGradRequest,
    ZeroGradResponse,
)
from tinkerbell.utils import get_host_and_port

APP = FastAPI()


class TinkerbellServiceDeployment:
    def __init__(
        self,
        server_url: str,
        max_wait_time: float = 300.0,
        clock_cycle: float = 10.0,
    ):
        self.server_url = server_url
        print(
            f"initializing training manager with max_wait_time: {max_wait_time} and clock_cycle: {clock_cycle}"
        )
        self.training_manager = TrainingManager(
            max_wait_time=max_wait_time,
            clock_cycle=clock_cycle,
        )

    @APP.post("/zero_grad")
    async def zero_grad(self, request: ZeroGradRequest) -> ZeroGradResponse:
        await self.training_manager.zero_grad(model_name=request.model_name)
        return ZeroGradResponse(
            model_name=request.model_name,
            message=f"Gradients zeroed for model {request.model_name}",
        )

    @APP.post("/optim_step")
    async def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        await self.training_manager.optim_step(
            model_name=request.model_name, optimizer_params=request.optimizer_params
        )
        return OptimStepResponse(
            model_name=request.model_name,
            message=f"Optimizer stepped for model {request.model_name}",
        )

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
            world_size=request.world_size,
            master_addr=request.master_addr,
            master_port=request.master_port,
            model_name=request.model_name,
            model_kwargs=request.model_kwargs,
            parallelize_plan=request.parallelize_plan,
            scheduler_params=request.scheduler_params,
            ray_worker_options=request.ray_worker_options,
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
    ) -> ForwardBackwardResponse:
        if not self.training_manager.running:
            await self.training_manager.start()
        print(f"Request: {request}")
        print(f"Request inputs: {request.inputs}, type: {type(request.inputs)}")
        print(f"Request targets: {request.targets}, type: {type(request.targets)}")
        print(f"Request forward_kwargs: {request.forward_kwargs}")
        print(f"Request return_logprobs: {request.return_logprobs}")
        remote_future: RemoteFuture = await self.training_manager.forward_backward(
            model_name=request.model_name,
            inputs=request.inputs,
            targets=request.targets,
            forward_kwargs=request.forward_kwargs,
            return_logprobs=request.return_logprobs,
        )
        return ForwardBackwardResponse(
            model_name=request.model_name,
            request_id=remote_future.request_id,
            loss=None,
            logprobs=None,
            outputs=None,
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

    @APP.post("/get_ray_actors")
    async def get_ray_actors(
        self,
    ) -> GetRayActorsResponse:
        actors = ray.util.list_named_actors()
        return GetRayActorsResponse(
            actor_names=actors,
        )


def deploy_service(
    server_url: str,
    max_wait_time: float = 300.0,
    clock_cycle: float = 10.0,
    **deployment_kwargs,
):
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
        deployment.bind(
            server_url=server_url,
            max_wait_time=max_wait_time,
            clock_cycle=clock_cycle,
        ),
    )

    return server_url


def deploy_on_modal(
    server_url: str,
    max_wait_time: float = 300.0,
    clock_cycle: float = 10.0,
    deploy_config: ModalDeployConfig = ModalDeployConfig(),
):
    """Deploy the TinkerbellService on Modal.

    Args:
        server_url: URL of the backend training server
        **deployment_kwargs: Optional Modal deployment configurations
            (e.g., num_replicas, ray_actor_options, autoscaling_config)
    """
    try:
        import os
        import sys

        import modal
        from modal import runner
    except ImportError:
        raise ImportError("Modal is not installed. Install it with: pip install modal")

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
            "nvidia/cuda:12.6.0-devel-ubuntu22.04",
            add_python=f"{sys.version_info.major}.{sys.version_info.minor}",
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
        deploy_service(
            server_url=server_url,
            max_wait_time=max_wait_time,
            clock_cycle=clock_cycle,
        )

    with modal.enable_output():
        # Deploy the app
        runner.deploy_app(app)
