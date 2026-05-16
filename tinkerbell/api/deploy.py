"""Deployment helpers for the Tinkerbell server."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def deploy_on_modal(
    server_url: str = "https://0.0.0.0:8000",
    gpu: str = "H100",
    num_gpus: int = 1,
    timeout: int = 86400,
    scaledown_window: int = 600,
    max_inputs: int = 100,
    max_containers: int = 1,
    memory_mb: int = 8192,
) -> str:
    try:
        import modal
        from modal import runner
    except ImportError as e:
        raise ImportError("Modal is not installed. pip install modal") from e

    app = modal.App(name="tinkerbell-service")
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
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "NCCL_DEBUG": "INFO",
                "TORCH_DISTRIBUTED_BACKEND": "nccl",
                "RAY_DEDUP_LOGS": "0",
            }
        )
        .add_local_python_source("tinkerbell")
    )
    volume = modal.Volume.from_name("tinkerbell-checkpoints", create_if_missing=True)

    @app.function(
        image=image,
        gpu=f"{gpu}:{num_gpus}",
        volumes={"/checkpoints": volume},
        timeout=timeout,
        scaledown_window=scaledown_window,
        max_containers=max_containers,
        memory=memory_mb,
        serialized=True,
        name="serve",
    )
    @modal.concurrent(max_inputs=max_inputs)
    @modal.web_server(8000, label="training-service")
    def serve_fn():
        from tinkerbell.api.server import deploy_service

        deploy_service(server_url=server_url)

    logger.info("Deploying Tinkerbell server on Modal...")
    with modal.enable_output():
        runner.deploy_app(app)
    return modal.Function.from_name("tinkerbell-service", "serve").get_web_url()


def deploy_service(*args, **kwargs):
    from tinkerbell.api.server import deploy_service as _deploy_service

    return _deploy_service(*args, **kwargs)


__all__ = ["deploy_on_modal", "deploy_service"]
