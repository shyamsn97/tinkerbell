# import asyncio
import asyncio
import logging
import uuid
from functools import wraps
from typing import Any, Dict

import ray
from fastapi import FastAPI, Request
from ray import serve

from tinkerbell.sampling.manager import SamplingManager
from tinkerbell.store import GlobalStore
from tinkerbell.training.manager import TrainingManager
from tinkerbell.types import (
    ActorStatusRequest,
    ActorStatusResponse,
    CreateSamplingActorRequest,
    CreateSamplingActorResponse,
    CreateTrainingActorsRequest,
    CreateTrainingActorsResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    HealthResponse,
    LoadCheckpointRequest,
    PollResultRequest,
    PollResultResponse,
    RemoteFuture,
    SampleRequest,
    SaveCheckpointRequest,
    ShutdownSamplingActorRequest,
)
from tinkerbell.types.data import TensorData
from tinkerbell.types.optimizer import (
    OptimStepRequest,
    ZeroGradRequest,
)
from tinkerbell.utils import get_host_and_port, model_to_dict

logger = logging.getLogger(__name__)


def returns_future(func):
    """
    Decorator that wraps an async method to:
    1. Generate a unique request_id
    2. Execute the method in a background task
    3. Store the result in global_store
    4. Return a RemoteFuture immediately

    The decorated function should return the result dict to be stored.
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> RemoteFuture:
        request_id = str(uuid.uuid4())

        async def _execute():
            try:
                result = await func(self, *args, **kwargs)
                await self.global_store.set_result.remote(
                    request_id=request_id, result=result
                )
            except Exception as e:
                logger.error(
                    f"Error in background task for {func.__name__}: {e}", exc_info=True
                )
                # Store the error result so the client can see what went wrong
                error_result = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "message": f"Error in {func.__name__}: {str(e)}",
                }
                await self.global_store.set_result.remote(
                    request_id=request_id, result=error_result
                )

        _ = asyncio.create_task(_execute())
        # Small yield to let the task start
        await asyncio.sleep(0.1)
        return RemoteFuture(request_id=request_id)

    return wrapper


APP = FastAPI()


class TinkerbellServiceDeployment:
    def __init__(
        self,
        server_url: str,
        max_wait_time: float = 600.0,
        clock_cycle: float = 10.0,
    ):
        self.server_url = server_url
        logger.info(
            f"initializing training manager with max_wait_time: {max_wait_time} and clock_cycle: {clock_cycle}"
        )
        self.global_store = self._create_global_store()
        self.training_manager = TrainingManager(
            max_wait_time=max_wait_time,
            clock_cycle=clock_cycle,
            global_store=self.global_store,
        )
        self.sampling_manager = SamplingManager(
            global_store=self.global_store,
        )

    def _create_global_store(self) -> Any:
        """Create the global store Ray actor."""
        return GlobalStore.options(
            num_gpus=0,
            get_if_exists=True,
            lifetime="detached",
            name="tinkerbell_global_state_manager",
            namespace="tinkerbell",
        ).remote()

    @APP.post("/poll_result")
    async def poll_result(
        self,
        http_request: Request,
    ) -> PollResultResponse:
        """
        Poll for result without blocking.
        Returns status: "pending", "completed", or "error"

        Query parameters:
            delete_after_retrieval: If true, delete the result after returning it (default: true)
        """
        # Manually parse the request body
        body = await http_request.json()
        logger.info(f"[poll_result] ENDPOINT HIT! Body: {body}")
        request = PollResultRequest(**body)

        # Check if we should delete after retrieval (default: true for backward compatibility)
        delete_after = body.get("delete_after_retrieval", True)

        logger.debug(f"[poll_result] Polling for request_id: {request.request_id}")
        results = await self.global_store.get_result.remote(
            request_id=request.request_id
        )
        logger.debug(
            f"[poll_result] Result for {request.request_id}: {results is not None}"
        )
        if results is None:
            return PollResultResponse(
                status="pending",
                request_id=request.request_id,
                result=None,
            )

        # Check if the result indicates an error
        if isinstance(results, dict) and results.get("success") is False:
            logger.error(
                f"[poll_result] Returning error result for request_id: {request.request_id}, error: {results.get('error')}"
            )
            # Clean up the error result from store if requested
            if delete_after:
                await self.global_store.delete_result.remote(
                    request_id=request.request_id
                )
                logger.debug(
                    f"[poll_result] Deleted error result for request_id: {request.request_id}"
                )
            return PollResultResponse(
                status="error",
                request_id=request.request_id,
                result=results,
                error=results.get("error", "Unknown error"),
            )

        logger.info(
            f"[poll_result] Returning completed result for request_id: {request.request_id}"
        )
        # Clean up the completed result from store if requested
        if delete_after:
            await self.global_store.delete_result.remote(request_id=request.request_id)
            logger.debug(
                f"[poll_result] Deleted completed result for request_id: {request.request_id}"
            )
        return PollResultResponse(
            status="completed",
            request_id=request.request_id,
            result=results,
        )

    @APP.post("/zero_grad")
    @returns_future
    async def zero_grad(self, request: ZeroGradRequest) -> RemoteFuture:
        await self.training_manager.zero_grad(model_name=request.model_id)
        return {
            "model_id": request.model_id,
            "message": f"Gradients zeroed for {request.model_id}",
        }

    @APP.post("/optim_step")
    @returns_future
    async def optim_step(self, request: OptimStepRequest) -> RemoteFuture:
        await self.training_manager.optim_step(
            model_name=request.model_id,
            adapter_name=request.adapter_name,
            optimizer_params=request.optimizer_params,
        )
        return {
            "model_id": request.model_id,
            "message": f"Optimizer stepped for {request.model_id}",
        }

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
            model_id=request.model_id,
            model_name=request.model_name,
            adapter_name=request.adapter_name,
            model_kwargs=request.model_kwargs,
            parallelize_plan=request.parallelize_plan,
            scheduler_params=request.scheduler_params,
            ray_worker_options=request.ray_worker_options,
            lora_config=request.lora_config,
            initialize_random_weights=request.initialize_random_weights,
        )
        return CreateTrainingActorsResponse(
            success=True,
            model_id=model_name,
            message=f"Training actors for {model_name} created...",
        )

    @APP.post("/save_checkpoint")
    @returns_future
    async def save_checkpoint(self, request: SaveCheckpointRequest) -> RemoteFuture:
        await self.training_manager.save_checkpoint(
            model_name=request.model_id,
            checkpoint_path=request.checkpoint_path,
            adapter_name=request.adapter_name,
        )
        return {
            "model_id": request.model_id,
            "success": True,
            "message": f"Checkpoint saved for {request.model_id}",
            "path": request.checkpoint_path,
        }

    @APP.post("/forward_backward")
    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> ForwardBackwardResponse:
        if not self.training_manager.running:
            await self.training_manager.start()
        remote_future: RemoteFuture = await self.training_manager.forward_backward(
            model_name=request.model_id,
            adapter_name=request.adapter_name,
            data=request.data,
            forward_kwargs=request.forward_kwargs,
            return_logprobs=request.return_logprobs,
        )
        return remote_future

    @APP.post("/get_actor_status")
    async def get_actor_status(
        self, request: ActorStatusRequest
    ) -> ActorStatusResponse:
        status = await self.training_manager.get_actor_status(
            model_name=request.model_id
        )
        return ActorStatusResponse(
            status=status.value,
            message=f"Actor status for {request.model_id}: {status.value}",
        )

    @APP.get("/get_store_keys")
    async def get_store_keys(self) -> Dict[str, Any]:
        """Get list of all keys from the global store."""
        try:
            keys = await self.global_store.get_keys.remote()
            return {"keys": keys}
        except Exception as e:
            logger.error(f"Error getting store keys: {e}", exc_info=True)
            return {"keys": [], "error": str(e)}

    @APP.post("/get_ray_actors")
    @returns_future
    async def get_ray_actors(
        self,
    ) -> RemoteFuture:
        actors = ray.util.list_named_actors(all_namespaces=True)
        # Handle both string and dict return formats from ray.util.list_named_actors()
        actor_names = []
        for actor in actors:
            if isinstance(actor, str):
                actor_names.append(actor)
            elif isinstance(actor, dict):
                actor_names.append(actor.get("name", str(actor)))
            else:
                actor_names.append(str(actor))
        result = {"actor_names": actor_names}
        return result

    @APP.post("/create_sampling_actor")
    async def create_sampling_actor(
        self, request: CreateSamplingActorRequest
    ) -> CreateSamplingActorResponse:
        key = self.sampling_manager.create_sampling_actor(
            model_id=request.model_id,
            model_name=request.model_name,
            tp_size=request.tp_size,
            engine_kwargs=request.engine_kwargs,
        )
        return CreateSamplingActorResponse(
            success=True, message=f"Sampling actor for {key} created..."
        )

    @APP.post("/get_sampling_actor_status")
    @returns_future
    async def get_sampling_actor_status(
        self,
        request: ActorStatusRequest,
    ) -> RemoteFuture:
        status = await self.sampling_manager.get_sampling_actor_status(request.model_id)
        result = {
            "status": status.value,
            "message": f"Sampling actor status for model {request.model_id} is {status.value}",
        }
        return result

    @APP.post("/sample")
    @returns_future
    async def sample(
        self,
        request: SampleRequest,
    ) -> RemoteFuture:
        sampling_actor = self.sampling_manager.get_sampling_actor(request.model_id)
        if sampling_actor is None:
            raise ValueError(f"Sampling actor for model {request.model_id} not found")

        request_dict = model_to_dict(request, exclude=["model_id"], exclude_none=True)

        # Convert TensorData to list format after model_to_dict
        if isinstance(request.input_ids, TensorData):
            request_dict["input_ids"] = request.input_ids.tolist()

        if isinstance(request.input_embeds, TensorData):
            request_dict["input_embeds"] = request.input_embeds.tolist()

        ref = sampling_actor.sample.remote(request_dict)
        sample_result = await ref
        result = {
            "outputs": sample_result.get("outputs", []),
            "logprobs": sample_result.get("logprobs"),
            "top_logprobs": sample_result.get("top_logprobs"),
            "output_token_ids": sample_result.get("output_token_ids"),
            "finish_reasons": sample_result.get("finish_reasons"),
            "meta_info": sample_result.get("meta_info"),
        }
        return result

    @APP.post("/load_checkpoint")
    @returns_future
    async def load_checkpoint(
        self,
        request: LoadCheckpointRequest,
    ) -> RemoteFuture:
        try:
            _ = await self.sampling_manager.load_checkpoint(
                model_id=request.model_id,
                checkpoint_path=request.checkpoint_path,
                pin_lora=request.pin_lora,
            )
            result = {
                "model_id": request.model_id,
                "success": True,
                "message": f"Checkpoint loading started from {request.checkpoint_path}. Use get_sampling_actor_status to check when ready.",
            }
        except Exception as e:
            result = {
                "model_id": request.model_id,
                "success": False,
                "message": f"Failed to start checkpoint loading: {str(e)}",
            }
        return result

    @APP.post("/shutdown_sampling_actor")
    @returns_future
    async def shutdown_sampling_actor(
        self,
        request: ShutdownSamplingActorRequest,
    ) -> RemoteFuture:
        try:
            _ = await self.sampling_manager.shutdown(model_id=request.model_id)
            result = {
                "model_id": request.model_id,
                "success": True,
                "message": f"Sampling actor for model {request.model_id} shut down successfully",
            }
        except Exception as e:
            result = {
                "model_id": request.model_id,
                "success": False,
                "message": f"Failed to shutdown sampling actor: {str(e)}",
            }
        return result


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

    # Check if server already exists
    print("Starting server deployment...")
    try:
        existing_function = modal.Function.from_name(
            "tinkerbell-service", "deploy_on_modal.<locals>.serve"
        )
        existing_url = existing_function.web_url
        logger.info(f"Server already exists at: {existing_url}")
        return existing_url
    except (modal.exception.NotFoundError, Exception):
        logger.info("No existing server found, deploying new one...")

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
        # .pip_install(
        #     "packaging",
        #     "ninja",
        #     "wheel",
        #     "setuptools",
        # )
        # .run_commands(
        #     "pip install flash-attn==2.8.3 --no-build-isolation",
        # )
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
            "peft",
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

    print("Deploying server on Modal...")
    with modal.enable_output():
        runner.deploy_app(app)

    return modal.Function.from_name(
        "tinkerbell-service", "deploy_on_modal.<locals>.serve"
    ).get_web_url()
