from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx  # noqa
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    sampling_params: Dict[str, Any] = Field(default_factory=dict)
    generate_kwargs: Dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    finish_reason: str
    tokens_generated: Optional[int] = None


class ForwardRequest(BaseModel):
    prompt: str
    return_logprobs: bool = True
    top_logprobs_num: int = 5


class ForwardResponse(BaseModel):
    logprobs: List[float]
    token_logprobs: List[List[float]]
    tokens: List[str]
    top_logprobs: Optional[List[dict]] = None
    prompt: str


class GetWeightRequest(BaseModel):
    weight_name: str
    truncate_size: int = 100


class ServerConfig(BaseModel):
    server_url: str
    model_name: str
    tokenizer: str | None = None


class SGLangServerConfig(ServerConfig):
    engine_kwargs: Dict[str, Any] = Field(default_factory=dict)


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

    @property
    def deployment_type(self) -> str:
        return "modal"


def get_host_and_port(server_url: str) -> tuple[str, int | None]:
    """
    Parse server URL to extract host and port.

    Args:
        server_url: URL in format "http://host:port", "host:port", "http://host", or "host"

    Returns:
        Tuple of (host, port) where port is None if not specified
    """
    from urllib.parse import urlparse

    # Add scheme if not present to help urlparse
    if not server_url.startswith(("http://", "https://")):
        server_url = "https://" + server_url

    parsed = urlparse(server_url)
    host = parsed.hostname or parsed.netloc.split(":")[0]
    port = parsed.port

    return host, port


class SGLangServer:
    def __init__(
        self,
        config: SGLangServerConfig,
    ):
        self.model_name = config.model_name
        self.tokenizer = config.tokenizer or self.model_name
        self.engine_kwargs = config.engine_kwargs
        self.app = FastAPI(title="SGLang Inference Server")
        self.engine = None
        self.host, self.port = get_host_and_port(config.server_url)

    def _setup_engine(self):
        """Initialize the SGLang async engine with the model."""
        from sglang import Engine  # noqa

        self.engine = Engine(
            model_path=self.model_name,
            tokenizer_path=self.tokenizer,
            **self.engine_kwargs,
        )

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize the SGLang async engine with the model."""
            import sglang
            from sglang import Engine  # noqa

            logger.info(f"SGLang version: {sglang.__version__}")
            if not self.engine:
                self._setup_engine()
            logger.info(f"SGLang Engine initialized with model: {self.model_name}")

        # @self.app.on_event("shutdown")
        # async def shutdown_event():
        #     if self.engine:
        #         await self.engine.shutdown()
        #         print("SGLang Engine shut down")

        @self.app.get("/health")
        async def health():
            import sglang

            logger.info(f"SGLang version: {sglang.__version__}")
            return {
                "status": "healthy" if self.engine else "initializing",
                "model": self.model_name,
            }

        @self.app.post("/batch_forward")
        async def batch_forward(requests: List[ForwardRequest]):
            """
            Batch forward pass to get logprobs for multiple prompts.
            """
            if not self.engine:
                return {"error": "Engine not initialized"}

            prompts = [req.prompt for req in requests]

            results = await self.engine.async_batch_forward(
                prompts,
                return_logprobs=requests[0].return_logprobs,
                top_logprobs_num=requests[0].top_logprobs_num,
            )

            return [
                ForwardResponse(
                    logprobs=result["logprobs"],
                    token_logprobs=result.get("token_logprobs", []),
                    tokens=result["tokens"],
                    top_logprobs=result.get("top_logprobs"),
                    prompt=prompts[i],
                )
                for i, result in enumerate(results)
            ]

        @self.app.post("/batch_generate")
        async def batch_generate(requests: BatchGenerateRequest):
            """Generate text completions for multiple prompts concurrently."""
            if not self.engine:
                return {"error": "Engine not initialized"}

            request_json = json.loads(requests.json())
            prompts = request_json.get("prompts")
            sampling_params = request_json.get("sampling_params")
            generate_kwargs = request_json.get("generate_kwargs")

            # if isinstance(sampling_params_list, dict):
            #     sampling_params_list = [sampling_params_list] * len(prompts)

            # Batch async generation
            print("SAMPLING PARAMS LIST", sampling_params)
            results = await self.engine.async_generate(
                prompts,
                sampling_params=sampling_params,
                **generate_kwargs,
            )
            logger.info(f"Results: {results}")
            print("RESULTS", results)

            return [
                GenerateResponse(
                    text=result["text"],
                    prompt=prompts[i],
                    finish_reason=result.get("finish_reason", "stop"),
                    tokens_generated=result.get("tokens_generated"),
                )
                for i, result in enumerate(results)
            ]

        @self.app.post("/get_weights_by_name")
        async def get_weights_by_name(request: GetWeightRequest):
            """
            Get a model weight by parameter name.

            Args:
                weight_name: The name of the weight parameter
                truncate_size: Maximum size to return (default: 100)
            """
            if not self.engine:
                return {"error": "Engine not initialized"}

            try:
                result = self.engine.get_weights_by_name(
                    name=request.weight_name, truncate_size=request.truncate_size
                )
                return result
            except Exception as e:
                logger.error(f"Error getting weight '{request.weight_name}': {str(e)}")
                return {"error": str(e)}

    def deploy(self, deploy_config: DeployConfig = DeployConfig()) -> str:
        """Deploy the SGLang server."""
        self.setup_routes()
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
        server_url = (
            f"http://{self.host}:{self.port}" if self.port else f"http://{self.host}"
        )
        return server_url


class ModalSGLangServer(SGLangServer):

    def deploy(
        self,
        deploy_config: DeployConfig = ModalDeployConfig(),
    ) -> None:
        """
        Deploy the training server to Modal.

        Returns:
            modal.Function: The deployed Modal function
        """

        try:
            import modal
            from modal import runner
        except ImportError:
            raise ImportError(
                "Modal is not installed. Install it with: pip install modal"
            )

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
        }

        app_name = "tinkerbell-sglang-inference"
        label = "sglang-model"
        app_name = f"{app_name}-{label}"
        # Create Modal app
        app = modal.App(name=app_name)

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
                "ray"
            )
            .env(env_variables)
        )
        image = image.add_local_python_source("tinkerbell")

        # Create a volume for model checkpoints if needed
        volume = modal.Volume.from_name(
            "tinkerbell-checkpoints", create_if_missing=True
        )

        # IMPORTANT: Capture these values as local variables to avoid serializing self
        model_name = self.model_name
        tokenizer = self.tokenizer
        host = self.host
        port = self.port
        engine_kwargs = self.engine_kwargs.copy()

        if deploy_config.num_gpus > 1:
            engine_kwargs["tp_size"] = deploy_config.num_gpus
            # Add NCCL backend specification

        # Remove the nested serve() function entirely and use this instead:
        @app.cls(
            image=image,
            gpu=f"{deploy_config.gpu}:{deploy_config.num_gpus}",
            volumes={"/checkpoints": volume},
            timeout=deploy_config.timeout,
            container_idle_timeout=deploy_config.container_idle_timeout,
            serialized=True,
        )
        @modal.concurrent(max_inputs=32)
        class SGLangService:
            @modal.enter()
            def initialize(self):
                """Initialize engine when container starts."""

                # logger.info("Initializing SGLang Engine in Modal container...")
                # self.engine = Engine(
                #     model_path=model_name,
                #     tokenizer_path=tokenizer,
                #     **engine_kwargs,
                # )
                # logger.info("Engine initialized successfully!")
                # Create FastAPI app with initialized engine
                config = SGLangServerConfig(
                    server_url=f"{host}:{port}" if port else host,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    engine_kwargs=engine_kwargs,
                )
                server = SGLangServer(config=config)
                server._setup_engine()
                server.setup_routes()
                self.app = server.app

            @modal.asgi_app(label=label)
            def serve(self):
                """Return the FastAPI app."""
                return self.app

        # # Define the Modal function
        # @app.function(
        #     image=image,
        #     gpu=f"{deploy_config.gpu}:{deploy_config.num_gpus}",  # Request GPUs
        #     volumes={"/checkpoints": volume},
        #     timeout=deploy_config.timeout,  # 24 hours
        #     serialized=deploy_config.serialized,
        #     container_idle_timeout=deploy_config.container_idle_timeout,  # 5 minutes
        # )
        # @modal.concurrent(max_inputs=32)
        # @modal.asgi_app(label=label)
        # def serve():
        #     """Serve the FastAPI app on Modal."""
        #     # Create the server instance INSIDE the Modal function
        #     # This prevents Modal from trying to serialize it
        #     config = SGLangServerConfig(
        #         server_url=f"{host}:{port}" if port else host,
        #         model_name=model_name,
        #         tokenizer=tokenizer,
        #         engine_kwargs=engine_kwargs,
        #     )
        #     server = SGLangServer(config=config)

        #     # Initialize engine IMMEDIATELY, not in startup event
        #     server._setup_engine()

        #     # Now set up routes (without the startup event doing initialization)
        #     server.setup_routes()
        #     return server.app

        with modal.enable_output():
            # Deploy the app
            runner.deploy_app(app)

        # Get the workspace name from Modal config
        workspace = modal.config._profile
        server_url = f"https://{workspace}--{label}.modal.run"

        # logger.info(f"Server deployed at: {server_url}")
        # logger.info(
        #     f"Triggering engine initialization via health check at {server_url}/health"
        # )

        # try:
        #     with httpx.Client(timeout=600.0) as client:
        #         response = client.get(f"{server_url}/health")
        #         if response.status_code == 200:
        #             logger.info(f"Engine initialized successfully: {response.json()}")
        # except (httpx.HTTPError, httpx.TimeoutException) as e:
        #     logger.warning(f"Could not verify engine initialization: {e}")

        return server_url


def wait_for_server(
    server_url: str,
    max_wait_time: int = 600,  # 10 minutes default
    retry_interval: int = 5,  # Check every 5 seconds
    endpoint: str = "/health",
) -> bool:
    """
    Wait for the server to be ready by polling the health endpoint.

    Args:
        server_url: Base URL of the server
        max_wait_time: Maximum time to wait in seconds
        retry_interval: Time between retry attempts in seconds
        endpoint: Health check endpoint to poll

    Returns:
        True if server is ready, False if timeout
    """

    start_time = time.time()
    attempt = 0
    health_url = f"{server_url}{endpoint}"

    logger.info(f"Waiting for server at {health_url} (max {max_wait_time}s)...")

    while time.time() - start_time < max_wait_time:
        attempt += 1
        elapsed = time.time() - start_time

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(health_url)

                if response.status_code == 200:
                    logger.info(
                        f"✓ Server ready after {elapsed:.1f}s "
                        f"({attempt} attempts): {response.json()}"
                    )
                    return True
                else:
                    logger.debug(
                        f"Attempt {attempt}: Server returned status {response.status_code}, "
                        f"waiting... ({elapsed:.1f}s elapsed)"
                    )

        except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as e:
            logger.debug(
                f"Attempt {attempt}: Connection failed ({type(e).__name__}), "
                f"waiting... ({elapsed:.1f}s elapsed)"
            )
        except Exception as e:
            logger.warning(
                f"Attempt {attempt}: Unexpected error: {e}, "
                f"waiting... ({elapsed:.1f}s elapsed)"
            )

        # Wait before next attempt
        time.sleep(retry_interval)

    # Timeout reached
    total_elapsed = time.time() - start_time
    logger.error(
        f"✗ Server not ready after {total_elapsed:.1f}s ({attempt} attempts). "
        f"Timeout reached."
    )
    return False
