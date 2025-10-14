from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False


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


class FastAPISGLangServer:
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        **engine_kwargs,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer or model_name
        self.host = host
        self.port = port
        self.engine_kwargs = engine_kwargs
        self.app = FastAPI(title="SGLang Inference Server")
        self.engine = None
        self.setup_routes()

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize the SGLang async engine with the model."""
            import sglang
            from sglang import Engine  # noqa

            print(f"SGLang version: {sglang.__version__}")
            self.engine = Engine(
                model_path=self.model_name,
                tokenizer_path=self.tokenizer,
                **self.engine_kwargs,
            )
            print(f"SGLang Engine initialized with model: {self.model_name}")

        # @self.app.on_event("shutdown")
        # async def shutdown_event():
        #     if self.engine:
        #         await self.engine.shutdown()
        #         print("SGLang Engine shut down")

        @self.app.get("/health")
        async def health():
            import sglang

            print(f"SGLang version: {sglang.__version__}")
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
        async def batch_generate(requests: List[GenerateRequest]):
            """Generate text completions for multiple prompts concurrently."""
            if not self.engine:
                return {"error": "Engine not initialized"}

            # Batch async generation
            prompts = [req.prompt for req in requests]
            sampling_params_list = [
                {
                    "max_new_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "stop": req.stop,
                }
                for req in requests
            ]

            results = await self.engine.async_generate(
                prompts, sampling_params=sampling_params_list
            )

            return [
                GenerateResponse(
                    text=result["text"],
                    prompt=prompts[i],
                    finish_reason=result.get("finish_reason", "stop"),
                    tokens_generated=result.get("tokens_generated"),
                )
                for i, result in enumerate(results)
            ]

    def deploy(self):
        """Start the FastAPI server."""
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


class SGLangServer:
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        **engine_kwargs,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer or model_name
        self.host = host
        self.port = port
        self.engine_kwargs = engine_kwargs
        self.server = None

    def deploy(self):
        if self.server is None:
            self.server = FastAPISGLangServer(host="0.0.0.0", port=8000)
        self.server.deploy()

    def deploy_to_modal(self, gpu="H100:1"):
        """
        Deploy the training server to Modal.

        Returns:
            modal.Function: The deployed Modal function
        """
        import os

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
            gpu=gpu,  # Request GPU
            volumes={"/checkpoints": volume},
            timeout=86400,  # 24 hours
            serialized=True,
            container_idle_timeout=300,  # 5 minutes
        )
        @modal.concurrent(max_inputs=24)
        @modal.asgi_app(label=label)
        def serve():
            """Serve the FastAPI app on Modal."""
            # Create the server instance INSIDE the Modal function
            # This prevents Modal from trying to serialize it
            server = FastAPISGLangServer(
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                host="0.0.0.0",
                port=8000,
                **self.engine_kwargs,
            )
            return server.app

        with modal.enable_output():
            # Deploy the app
            runner.deploy_app(app)
