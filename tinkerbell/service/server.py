import asyncio
from typing import Dict

import ray
from fastapi import HTTPException
from ray import serve

from tinkerbell.actors.training import TrainingActor
from tinkerbell.service.base import APP, ServiceBackend
from tinkerbell.service.models import (
    ActorGroup,
    ActorStatusRequest,
    ActorStatusResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    ForwardRequest,
    ForwardResponse,
    HealthResponse,
    ModalDeployConfig,
)
from tinkerbell.utils import get_host_and_port


class TinkerbellServiceBackend(ServiceBackend):
    def __init__(self, server_url: str | None = None):
        self.actor_groups: Dict[str, ActorGroup] = {}
        super().__init__(server_url)

    async def forward(self, request: ForwardRequest) -> ForwardResponse:
        """Forward pass through the model and compute the loss."""
        # model_name = request.model_name
        return None

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> ForwardBackwardResponse:
        """Forward and backward pass through the model."""

        model_name = request.model_name
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )

        actor_group = self.actor_groups[model_name]
        # Poll until actors are ready
        while True:
            if actor_group.status == "ready":
                break
            _ = await self.get_actor_status(ActorStatusRequest(model_name=model_name))
            await asyncio.sleep(1.0)  # Wait 100ms before checking again

        # Now that actors are ready, perform forward_backward
        workers = actor_group.workers

        inputs = request.inputs
        forward_kwargs = request.forward_kwargs

        # Call forward_backward on all workers
        refs = [
            worker.forward_backward.remote(inputs, **forward_kwargs)
            for worker in workers
        ]
        losses = await asyncio.gather(*refs)

        # Assuming we want the result from the first worker or some aggregation
        rank0_loss = [loss for loss in losses if loss is not None][0]
        return ForwardBackwardResponse(loss=rank0_loss)

    async def create_training_actors(
        self, request: CreateTrainingActorsRequest
    ) -> CreateTrainingActorsResponse:
        """Create a training worker for the given model name."""
        num_gpus = 1  # num gpus per worker is 1 for tensor parallelism

        ray_training_actor = ray.remote(TrainingActor).options(
            num_gpus=num_gpus, **request.ray_worker_options
        )
        workers = []
        for rank in range(request.world_size):
            workers.append(
                ray_training_actor.remote(
                    rank=rank,
                    world_size=request.world_size,
                    master_addr=request.master_addr,
                    master_port=request.master_port,
                    model_name=request.model_name,
                    model_kwargs=request.model_kwargs,
                    parallelize_plan=request.parallelize_plan,
                    optimizer_params=request.optimizer_params,
                    scheduler_params=request.scheduler_params,
                )
            )
        # Trigger setup but don't wait
        setup_refs = [worker.setup.remote() for worker in workers]

        # Store for later querying
        self.actor_groups[request.model_name] = ActorGroup(
            workers=workers,
            setup_refs=setup_refs,
            status="initializing",
            request=request,
        )

        return CreateTrainingActorsResponse(
            success=True,
            model_name=request.model_name,
            message=f"Training actors created for model {request.model_name}",
        )

    async def get_actor_status(
        self, request: ActorStatusRequest
    ) -> ActorStatusResponse:
        """Check if training actors are ready."""
        model_name = request.model_name
        if model_name not in self.actor_groups:
            raise HTTPException(
                status_code=404, detail=f"Group not found for model {model_name}"
            )

        group: ActorGroup = self.actor_groups[model_name]
        setup_refs = group.setup_refs

        # Non-blocking check
        ready, _ = ray.wait(setup_refs, num_returns=len(setup_refs), timeout=1)

        if len(ready) == len(setup_refs):
            group.status = "ready"
            _ = ray.get(setup_refs)
            return ActorStatusResponse(
                status="ready",
                message=f"Training actors for model {model_name} are ready",
            )
        else:
            return ActorStatusResponse(
                status="initializing",
                message=f"Training actors for model {model_name} are initializing",
            )


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

    # Create FastAPI app and set up routes
    # app = FastAPI()

    # Apply any custom deployment options if provided

    deployment_kwargs["ray_actor_options"] = {"num_gpus": 0}

    @serve.deployment(**deployment_kwargs)
    @serve.ingress(APP)
    class TinkerbellServiceDeployment:
        def __init__(self, server_url: str):
            self.backend = TinkerbellServiceBackend(server_url)

        @APP.get("/health")
        async def health(self) -> HealthResponse:
            return HealthResponse(
                status=self.backend.health_status,
                name=self.backend.name,
            )

        @APP.post("/create_training_actors")
        async def create_training_actors(
            self,
            request: CreateTrainingActorsRequest,
        ) -> CreateTrainingActorsResponse:
            return await self.backend.create_training_actors(request)

        @APP.post("/forward")
        async def forward(self, request: ForwardRequest) -> ForwardResponse:
            return await self.backend.forward(request)

        @APP.post("/forward_backward")
        async def forward_backward(
            self,
            request: ForwardBackwardRequest,
        ) -> ForwardBackwardResponse:
            return await self.backend.forward_backward(request)

        @APP.post("/get_actor_status")
        async def get_actor_status(
            self,
            request: ActorStatusRequest,
        ) -> ActorStatusResponse:
            return await self.backend.get_actor_status(request)

    serve.run(
        TinkerbellServiceDeployment.bind(server_url),
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
