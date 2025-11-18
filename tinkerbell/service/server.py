# import asyncio
import logging
from typing import Any, Dict

import ray
from fastapi import FastAPI
from ray import serve

from tinkerbell.inference.manager import InferenceManager
from tinkerbell.training.manager import TrainingManager
from tinkerbell.types import (
    ActorStatusRequest,
    ActorStatusResponse,
    CreateInferenceActorRequest,
    CreateInferenceActorResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GenerateRequest,
    GenerateResponse,
    GetRayActorsResponse,
    HealthResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    RemoteFuture,
    SaveCheckpointRequest,
    SaveCheckpointResponse,
    ShutdownInferenceActorRequest,
    ShutdownInferenceActorResponse,
)
from tinkerbell.types.optimizer import (
    OptimStepRequest,
    OptimStepResponse,
    ZeroGradRequest,
    ZeroGradResponse,
)
from tinkerbell.utils import get_host_and_port, model_to_dict

logger = logging.getLogger(__name__)

APP = FastAPI()


class TinkerbellServiceDeployment:
    def __init__(
        self,
        server_url: str,
        max_wait_time: float = 600.0,
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
        self.inference_manager = InferenceManager()

    @APP.post("/zero_grad")
    async def zero_grad(self, request: ZeroGradRequest) -> ZeroGradResponse:
        await self.training_manager.zero_grad(model_id=request.model_id)
        return ZeroGradResponse(
            model_id=request.model_id,
            message=f"Gradients zeroed for model {request.model_id}",
        )

    @APP.post("/optim_step")
    async def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        await self.training_manager.optim_step(
            model_id=request.model_id, optimizer_params=request.optimizer_params
        )
        return OptimStepResponse(
            model_id=request.model_id,
            message=f"Optimizer stepped for model {request.model_id}",
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
        model_id = await self.training_manager.create_training_actors(
            world_size=request.world_size,
            model_id=request.model_id,
            model_kwargs=request.model_kwargs,
            parallelize_plan=request.parallelize_plan,
            scheduler_params=request.scheduler_params,
            ray_worker_options=request.ray_worker_options,
            initialize_random_weights=request.initialize_random_weights,
        )
        return CreateTrainingActorsResponse(
            success=True,
            model_id=model_id,
            message=f"Training actors for model {model_id} created...",
        )

    @APP.post("/save_checkpoint")
    async def save_checkpoint(
        self, request: SaveCheckpointRequest
    ) -> SaveCheckpointResponse:
        await self.training_manager.save_checkpoint(
            model_id=request.model_id, checkpoint_path=request.checkpoint_path
        )
        return SaveCheckpointResponse(
            model_id=request.model_id,
            success=True,
            message=f"Checkpoint saved for model {request.model_id}",
        )

    @APP.post("/forward_backward")
    async def forward_backward(
        self,
        request: ForwardBackwardRequest,
    ) -> ForwardBackwardResponse:
        if not self.training_manager.running:
            await self.training_manager.start()
        print(f"Request: {request}")
        print(f"Request data: {request.data}, type: {type(request.data)}")
        print(f"Request forward_kwargs: {request.forward_kwargs}")
        print(f"Request return_logprobs: {request.return_logprobs}")
        remote_future: RemoteFuture = await self.training_manager.forward_backward(
            model_id=request.model_id,
            data=request.data,
            forward_kwargs=request.forward_kwargs,
            return_logprobs=request.return_logprobs,
        )
        return ForwardBackwardResponse(
            model_id=request.model_id,
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
        status = await self.training_manager.get_actor_status(request.model_id)
        return ActorStatusResponse(
            status=status.value,
            message=f"Actor status for model {request.model_id} is {status.value}",
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

    @APP.post("/create_inference_actor")
    async def create_inference_actor(
        self,
        request: CreateInferenceActorRequest,
    ) -> CreateInferenceActorResponse:
        _ = self.inference_manager.create_inference_actor(
            model_id=request.model_id,
            tp_size=request.tp_size,
            engine_kwargs=request.engine_kwargs,
        )
        return CreateInferenceActorResponse(
            success=True,
            message=f"Inference actor for model {request.model_id} created...",
        )

    @APP.post("/get_inference_actor_status")
    async def get_inference_actor_status(
        self,
        request: ActorStatusRequest,
    ) -> ActorStatusResponse:
        status = await self.inference_manager.get_inference_actor_status(
            request.model_id
        )
        return ActorStatusResponse(
            status=status.value,
            message=f"Inference actor status for model {request.model_id} is {status.value}",
        )

    @APP.post("/generate")
    async def generate(
        self,
        request: GenerateRequest,
    ) -> GenerateResponse:
        inference_actor = self.inference_manager.get_inference_actor(request.model_id)
        if inference_actor is None:
            raise ValueError(f"Inference actor for model {request.model_id} not found")
        request_dict = model_to_dict(request, exclude=["model_id"], exclude_none=True)
        ref = inference_actor.generate.remote(request_dict)
        outputs = await ref
        return GenerateResponse(
            outputs=outputs,
        )

    @APP.post("/load_checkpoint")
    async def load_checkpoint(
        self,
        request: LoadCheckpointRequest,
    ) -> LoadCheckpointResponse:
        try:
            _ = await self.inference_manager.load_checkpoint(
                model_id=request.model_id,
                checkpoint_path=request.checkpoint_path,
            )
            return LoadCheckpointResponse(
                model_id=request.model_id,
                success=True,
                message=f"Checkpoint loading started from {request.checkpoint_path}. Use get_inference_actor_status to check when ready.",
            )
        except Exception as e:
            return LoadCheckpointResponse(
                model_id=request.model_id,
                success=False,
                message=f"Failed to start checkpoint loading: {str(e)}",
            )

    @APP.post("/shutdown_inference_actor")
    async def shutdown_inference_actor(
        self,
        request: ShutdownInferenceActorRequest,
    ) -> ShutdownInferenceActorResponse:
        try:
            _ = await self.inference_manager.shutdown(model_id=request.model_id)
            return ShutdownInferenceActorResponse(
                model_id=request.model_id,
                success=True,
                message=f"Inference actor for model {request.model_id} shut down successfully",
            )
        except Exception as e:
            return ShutdownInferenceActorResponse(
                model_id=request.model_id,
                success=False,
                message=f"Failed to shutdown inference actor: {str(e)}",
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
        # Don't specify num_gpus - let Ray auto-detect all GPUs
        ray.init(namespace="tinkerbell")

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
    server_url: str = "https://0.0.0.0:8000",
    max_wait_time: float = 300.0,
    clock_cycle: float = 10.0,
    gpu: str = "H100",
    num_gpus: int = 1,
    timeout: int = 86400,
    container_idle_timeout: int = 600,
    max_inputs: int = 100,
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
        gpu=f"{gpu}:{num_gpus}",
        volumes={"/checkpoints": volume},
        timeout=timeout,
        container_idle_timeout=container_idle_timeout,
        serialized=True,
    )
    @modal.concurrent(max_inputs=max_inputs)
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
        runner.deploy_app(app)

    return modal.Function.from_name(
        "tinkerbell-service", "deploy_on_modal.<locals>.serve"
    ).get_web_url()
