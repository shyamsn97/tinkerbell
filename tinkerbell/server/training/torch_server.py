"""Utility functions for efficient tensor serialization/deserialization."""

import io

# import threading
from typing import Any

# import httpx
# import requests
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse

LOCAL_PACKAGE = "tinkerbell"


def serialize_tensor(obj: Any) -> bytes:
    """Serialize a tensor or nested structure of tensors to bytes.
    Args:
        obj: A torch tensor, list, dict, or nested structure containing tensors
    Returns:
        bytes: Serialized representation
    """
    buffer = io.BytesIO()
    torch.save(obj, buffer, _use_new_zipfile_serialization=False)
    buffer.seek(0)
    return buffer.read()


def deserialize_tensor(data: bytes) -> Any:
    """Deserialize bytes back to tensor or nested structure.

    Args:
        data: Serialized bytes from serialize_tensor
    Returns:
        The deserialized tensor or nested structure
    """
    buffer = io.BytesIO(data)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu")


def serialize_payload(data: list[Any], **kwargs) -> bytes:
    """Serialize a complete payload including data and additional parameters.

    Args:
        data: List of data points (can contain tensors)
        loss_fn: Optional loss function or serializable representation
        **kwargs: Additional parameters to serialize

    Returns:
        bytes: Serialized payload
    """
    payload = {"data": data, **kwargs}
    return serialize_tensor(payload)


def deserialize_payload(data: bytes) -> dict:
    """Deserialize a complete payload.

    Args:
        data: Serialized bytes

    Returns:
        dict: Deserialized payload with 'data', 'loss_fn', and other fields
    """
    return deserialize_tensor(data)


class FastAPITorchServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.model = None  # Model stored on server
        self.setup_routes()
        self.modal_app = None
        self.modal_image = None

    def setup_routes(self):
        @self.app.post("/multiply")
        async def multiply(request: Request):
            """Forward pass endpoint."""
            body = await request.body()
            payload = deserialize_payload(body)
            data = payload["data"]
            value = payload["value"]
            out = data * value
            serialized_result = serialize_tensor(out)
            return FastAPIResponse(
                content=serialized_result, media_type="application/octet-stream"
            )

        @self.app.post("/initialize_model")
        async def initialize_model(request: Request):
            """Initialize a PyTorch linear model on the server."""
            body = await request.body()
            payload = deserialize_payload(body)
            input_dim = payload.get("input_dim", 10)
            output_dim = payload.get("output_dim", 5)

            # Create model on the server
            self.model = nn.Linear(input_dim, output_dim)
            self.model.eval()  # Set to evaluation mode

            return {
                "status": "success",
                "message": f"Model initialized with input_dim={input_dim}, output_dim={output_dim}",
                "model_params": sum(p.numel() for p in self.model.parameters()),
            }

        @self.app.post("/forward")
        async def forward(request: Request):
            """Forward pass through the model stored on server."""
            if self.model is None:
                return FastAPIResponse(
                    content=b'{"error": "Model not initialized. Call /initialize_model first."}',
                    status_code=400,
                    media_type="application/json",
                )

            body = await request.body()
            payload = deserialize_payload(body)
            input_tensor = payload["data"]

            # Ensure input is a tensor
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor)

            # Forward pass on server
            with torch.no_grad():
                output = self.model(input_tensor)

            serialized_result = serialize_tensor(output)
            return FastAPIResponse(
                content=serialized_result, media_type="application/octet-stream"
            )

        @self.app.get("/model_info")
        async def model_info():
            """Get information about the current model."""
            if self.model is None:
                return {"status": "no_model", "message": "No model initialized"}

            return {
                "status": "model_loaded",
                "input_features": self.model.in_features,
                "output_features": self.model.out_features,
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "weight_shape": list(self.model.weight.shape),
                "bias_shape": list(self.model.bias.shape),
            }

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

    # def deploy(self, config: dict | None = None) -> Any:
    #     """Deploy the server.

    #     Args:
    #         config: Optional configuration dictionary (can override host/port)

    #     Returns:
    #         Server configuration
    #     """
    #     if config:
    #         host = config.get('host', self.host)
    #         port = config.get('port', self.port)
    #     else:
    #         host = self.host
    #         port = self.port

    #     print(f"Starting Torch Training Server on {host}:{port}")

    #     self.server_thread = threading.Thread(
    #         target=uvicorn.run,
    #         args=(self.app,),
    #         kwargs={"host": host, "port": port},
    #         daemon=True  # Thread will close when main program exits
    #     )
    #     self.server_thread.start()

    #     return {"host": host, "port": port}


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

        # Create Modal app
        app = modal.App(name="tinkerbell-torch-training")

        # Define Modal image with required dependencies
        image = modal.Image.debian_slim().uv_pip_install(
            "torch",
            "numpy",
            "fastapi",
            "uvicorn",
            "pydantic",
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
            allow_concurrent_inputs=10,
            serialized=True,
        )
        @modal.asgi_app()
        def serve():
            """Serve the FastAPI app on Modal."""
            # Create the server instance INSIDE the Modal function
            # This prevents Modal from trying to serialize it
            server = FastAPITorchServer(host="0.0.0.0", port=8000)
            return server.app

        # Deploy the app
        runner.deploy_app(app)
