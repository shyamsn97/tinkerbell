"""Ray Serve + Modal deployment for the HTTP API.

`deploy_service` runs locally; `deploy_on_modal` builds a Modal image and
binds the Serve deployment under the `serve` function.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import ray
from ray import serve

from tinkerbell.api.routes import APP, TinkerbellAPI

logger = logging.getLogger(__name__)


def deploy_service(
    server_url: str,
    max_wait_time: float = 300.0,
    clock_cycle: float = 2.0,
    **deployment_kwargs: Any,
) -> str:
    """Deploy the gateway via Ray Serve.

    Always binds Serve's HTTP proxy on 0.0.0.0:8000 (required by Modal's
    `@modal.web_server(8000)` wrapper).
    """
    if not ray.is_initialized():
        ray.init(namespace="tinkerbell")

    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})

    deployment_kwargs["ray_actor_options"] = {"num_gpus": 0}
    deployment = serve.deployment(**deployment_kwargs)(
        serve.ingress(APP)(TinkerbellAPI)
    )
    serve.run(
        deployment.bind(
            server_url=server_url,
            max_wait_time=max_wait_time,
            clock_cycle=clock_cycle,
        )
    )
    return server_url


def deploy_on_modal(
    server_url: str = "https://0.0.0.0:8000",
    max_wait_time: float = 300.0,
    clock_cycle: float = 2.0,
    gpu: str = "H100",
    num_gpus: int = 1,
    timeout: int = 86400,
    scaledown_window: int = 600,
    max_inputs: int = 1000,
) -> str:
    import os

    try:
        import modal
        from modal import runner
    except ImportError as e:
        raise ImportError("Modal is not installed. pip install modal") from e

    app = modal.App(name="tinkerbell-service")
    env_variables = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "NCCL_DEBUG": "INFO",
        "TORCH_DISTRIBUTED_BACKEND": "nccl",
        "RAY_DEDUP_LOGS": "0",
    }
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.0-devel-ubuntu22.04",
            add_python=f"{sys.version_info.major}.{sys.version_info.minor}",
        )
        .apt_install("libnuma-dev", "build-essential", "clang")
        .env({"CUDA_HOME": "/usr/local/cuda"})
        .pip_install(
            "torch==2.4.0", extra_index_url="https://download.pytorch.org/whl/cu126"
        )
        .uv_pip_install(
            "pybase64",
            "zmq",
            "xformers",
            "transformers",
            "numpy",
            "fastapi",
            "uvicorn",
            "pydantic>=2.0",
            "cloudpickle",
            "dill",
            "flashinfer-python",
            "sglang[all]>=0.5.3",
            "sgl-kernel",
            "huggingface_hub",
            "hf_transfer",
            "ray",
            "ray[serve]",
            "peft",
            "tinker",
        )
        .env(env_variables)
        .add_local_python_source("tinkerbell")
    )
    volume = modal.Volume.from_name("tinkerbell-checkpoints", create_if_missing=True)

    @app.function(
        image=image,
        gpu=f"{gpu}:{num_gpus}",
        volumes={"/checkpoints": volume},
        timeout=timeout,
        scaledown_window=scaledown_window,
        serialized=True,
        name="serve",
    )
    @modal.concurrent(max_inputs=max_inputs)
    @modal.web_server(8000, label="training-service")
    def serve_fn():  # noqa: ARG001 — modal uses the name at runtime
        deploy_service(
            server_url=server_url,
            max_wait_time=max_wait_time,
            clock_cycle=clock_cycle,
        )

    logger.info("Deploying tinkerbell API on Modal...")
    with modal.enable_output():
        runner.deploy_app(app)
    return modal.Function.from_name("tinkerbell-service", "serve").get_web_url()
