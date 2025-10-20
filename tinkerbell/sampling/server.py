from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BatchGenerateRequest(BaseModel):
    prompts: List[str]
    sampling_params: Dict[str, Any] = Field(default_factory=dict)
    engine_kwargs: Dict[str, Any] = Field(default_factory=dict)


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


class SGLangServerConfig(BaseModel):
    model_name: str
    tokenizer: Optional[str] = None
    engine_kwargs: Dict[str, Any] = Field(default_factory=dict)


class DeployConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def deployment_type(self) -> str:
        return "local"


class ModalDeployConfig(DeployConfig):
    gpu: str = "H100:1"
    timeout: int = 86400
    serialized: bool = True
    container_idle_timeout: int = 300

    @property
    def deployment_type(self) -> str:
        return "modal"


class SGLangServer:
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        engine_kwargs: Dict[str, Any] = Field(default_factory=dict),
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer or model_name
        self.host = host
        self.port = port
        self.engine_kwargs = engine_kwargs
        self.app = FastAPI(title="SGLang Inference Server")
        self.engine = None

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize the SGLang async engine with the model."""
            import sglang
            from sglang import Engine  # noqa

            logger.info(f"SGLang version: {sglang.__version__}")
            self.engine = Engine(
                model_path=self.model_name,
                tokenizer_path=self.tokenizer,
                **self.engine_kwargs,
            )
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

    def deploy(self, deploy_config: DeployConfig = DeployConfig()) -> None:
        """Deploy the SGLang server."""
        self.setup_routes()
        uvicorn.run(
            self.app, host=deploy_config.host, port=deploy_config.port, log_level="info"
        )


class ModalSGLangServer(SGLangServer):

    def deploy(
        self,
        deploy_config: DeployConfig = DeployConfig(),
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
            )
            .env(env_variables)
        )
        image = image.add_local_python_source("tinkerbell")

        # Create a volume for model checkpoints if needed
        volume = modal.Volume.from_name(
            "tinkerbell-checkpoints", create_if_missing=True
        )

        # Define the Modal function
        @app.function(
            image=image,
            gpu=deploy_config.gpu,  # Request GPU
            volumes={"/checkpoints": volume},
            timeout=deploy_config.timeout,  # 24 hours
            serialized=deploy_config.serialized,
            container_idle_timeout=deploy_config.container_idle_timeout,  # 5 minutes
        )
        @modal.concurrent(max_inputs=24)
        @modal.asgi_app(label=label)
        def serve():
            """Serve the FastAPI app on Modal."""
            # Create the server instance INSIDE the Modal function
            # This prevents Modal from trying to serialize it
            server = SGLangServer(
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                host=deploy_config.host,
                port=deploy_config.port,
                engine_kwargs=self.engine_kwargs,
            )
            server.setup_routes()
            return server.app

        with modal.enable_output():
            # Deploy the app
            runner.deploy_app(app)
