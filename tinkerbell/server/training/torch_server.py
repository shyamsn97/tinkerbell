"""Utility functions for efficient tensor serialization/deserialization."""

# import io

# import threading
# from typing import Any

# from typing import Type

# from typing import Any, Dict

import dill

# import httpx
# import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse

from tinkerbell.utils import serialize_tensor

LOCAL_PACKAGE = "tinkerbell"


class FastAPITorchServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.registered_modules = {}  # Store registered module classes
        self.initialized_models = {}  # Store initialized models
        self.setup_routes()
        self.modal_app = None
        self.modal_image = None

    def register_module(self, name: str, module_class: type[nn.Module]):
        """Register a custom nn.Module class that can be instantiated via API.

        Args:
            name: Name to identify this module type
            module_class: The nn.Module class (not an instance)
        """
        if not issubclass(module_class, nn.Module):
            raise ValueError(f"{module_class} must be a subclass of nn.Module")
        self.registered_modules[name] = module_class
        print(f"Registered module: {name} -> {module_class.__name__}")

    def setup_routes(self):

        @self.app.post("/initialize_model")
        async def initialize_model(request: Request):
            """Initialize a PyTorch model on the server."""
            data = await request.json()
            name = data.get("name")
            config = data.get("config", {})

            # Create model on the server
            if name not in self.registered_modules:
                # Use registered custom module
                return FastAPIResponse(
                    content=f'{{"error": "Module type {name} not registered"}}'.encode(),
                    status_code=400,
                    media_type="application/json",
                )

            module_class = self.registered_modules[name]
            self.initialized_models[name] = module_class(**config)

            return {
                "status": "success",
                "message": f"Model {name} initialized with config={config}",
                "model_params": sum(
                    p.numel() for p in self.initialized_models[name].parameters()
                ),
            }

        @self.app.post("/forward")
        async def forward(request: Request):
            """Forward pass through the model stored on server."""
            data = await request.json()
            name = data.get("name")

            if self.initialized_models.get(name, None) is None:
                return FastAPIResponse(
                    content=b'{"error": "Model not initialized. Call /initialize_model first."}',
                    status_code=400,
                    media_type="application/json",
                )

            input_tensor = data.get("data")

            # Ensure input is a tensor
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor)

            # Forward pass on server
            output = self.initialized_models[name](input_tensor)

            serialized_result = serialize_tensor(output)
            return FastAPIResponse(
                content=serialized_result, media_type="application/octet-stream"
            )

        @self.app.get("/registered_modules")
        async def list_registered_modules():
            """List all registered module types."""
            return {
                "registered_modules": list(self.registered_modules.keys()),
                "module_details": {
                    name: cls.__name__ for name, cls in self.registered_modules.items()
                },
            }

        @self.app.post("/register_module")
        async def register_module(request: Request):
            """
            Register a PyTorch module class with the server.
            Expects pickled dict with 'name' and 'module_class_bytes' keys.
            """

            body_bytes = await request.body()
            data = dill.loads(body_bytes)  # or use deserialize_tensor if appropriate

            name = data.get("name")
            module_class_bytes = data.get("module_class")

            if not name:
                return FastAPIResponse(
                    content=b'{"error": "module name is required"}',
                    status_code=400,
                    media_type="application/json",
                )

            module_class = dill.loads(module_class_bytes)
            self.registered_modules[name] = module_class

            print(f"Registered module: {name}")
            return {"status": "success", "module_name": name}

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

    def start(self):
        """Start the server."""
        host = self.host
        port = self.port
        print(f"Starting Torch Training Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


class TorchServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port

    def start(self):
        self.server.start()

    def deploy(self):
        self.server.deploy()

    def deploy_to_modal(self):
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

        app_name = "tinkerbell-torch-training"
        label = "torch-nn-model"
        app_name = f"{app_name}-{label}"
        # Create Modal app
        app = modal.App(name=app_name)

        # Define Modal image with required dependencies
        image = modal.Image.debian_slim().uv_pip_install(
            "torch",
            "numpy",
            "fastapi",
            "uvicorn",
            "pydantic",
            "cloudpickle",
            "dill",
        )
        image = image.add_local_python_source(LOCAL_PACKAGE)

        # Create a volume for model checkpoints if needed
        volume = modal.Volume.from_name(
            "tinkerbell-checkpoints", create_if_missing=True
        )

        # Define the Modal function
        @app.function(
            image=image,
            gpu="any",  # Request GPU
            volumes={"/checkpoints": volume},
            timeout=86400,  # 24 hours
            serialized=True,
        )
        @modal.concurrent(max_inputs=24)
        @modal.asgi_app(label=label)
        def serve():
            """Serve the FastAPI app on Modal."""
            # Create the server instance INSIDE the Modal function
            # This prevents Modal from trying to serialize it
            server = FastAPITorchServer(host="0.0.0.0", port=8000)
            return server.app

        # Deploy the app
        runner.deploy_app(app)
