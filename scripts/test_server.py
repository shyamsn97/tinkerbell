from typing import Any
import httpx
from tinkerbell.utils import serialize_payload, deserialize_tensor, serialize_class
import torch
import torch.nn as nn

class TorchClient:

    def __init__(self, host: str = "localhost", port: int = 8000, base_url: str | None = None, timeout: float = 30.0):
        """Initialize the torch training client.

        Args:
            base_url: Base URL of the training server
            timeout: Request timeout in seconds
        """
        base_url = base_url or f"http://{host}:{port}"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def register_module(self, name: str, module_class: type[nn.Module]):
        """Register a custom nn.Module class that can be instantiated via API."""
        import dill
        payload = dill.dumps(module_class)
        payload = {
            "name": name,
            "module_class": payload,
        }
        # payload = serialize_class(module_class)
        return self.client.post(f"{self.base_url}/register_module", content=serialize_payload(payload), headers={"Content-Type": "application/octet-stream"})

    def list_registered_modules(self) -> Any:
        return self.client.get(f"{self.base_url}/registered_modules")

    def multiply(self, data: list[Any], value: float) -> Any:
        payload = serialize_payload(data, value=value)
        return self.client.post(f"{self.base_url}/multiply", content=payload, headers={"Content-Type": "application/octet-stream"})

    def initialize_model(self, config: dict) -> Any:
        """Initialize a PyTorch linear model on the server."""
        payload = serialize_payload([], config=config)
        response = self.client.post(
            f"{self.base_url}/initialize_model",
            content=payload,
            headers={"Content-Type": "application/octet-stream"}
        )
        return response.json()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model on the server."""
        payload = serialize_payload(data)
        response = self.client.post(
            f"{self.base_url}/forward",
            content=payload,
            headers={"Content-Type": "application/octet-stream"}
        )
        return deserialize_tensor(response.content)

    def list_registered_modules(self) -> dict:
        """Get information about the current model on the server."""
        response = self.client.get(f"{self.base_url}/registered_modules")
        return response.json()

    def health_check(self) -> Any:
        return self.client.get(f"{self.base_url}/health")

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Starting Torch Server Tests")
    print("=" * 60)

    base_url = "https://jesterlabs--torch-nn-model.modal.run"

    # Start server
    # server = TorchServer()
    # config = server.deploy()
    # print(f"\n✓ Server started at {config['host']}:{config['port']}")

    # Give server time to start
    # time.sleep(2)

    # Initialize client
    client = TorchClient(base_url=base_url, timeout=10.0)

    # Call health check with requests
    print(f"Health check response: {client.health_check()}")

    # Test 3: Initialize model
    # print("\n" + "=" * 60)
    # print("Test 3: Initialize Model")
    # print("=" * 60)
    # init_response = client.initialize_model(config={"input_dim": 10, "output_dim": 5})
    # print(f"Model initialization response: {init_response}")

    # Test 4: Model info
    print("\n" + "=" * 60)
    print("Test 4: Model Info")
    print("=" * 60)
    info = client.list_registered_modules()
    print(f"Model info: {info}")

    # Test 5: Register module
    print("\n" + "=" * 60)
    print("Test 5: Register Module")
    print("=" * 60)

    print(f"Before registering module")
    print(client.list_registered_modules())
    class CustomLinear(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.linear(x)) * 250.0
    print(f"CustomLinear: {CustomLinear}")
    register_response = client.register_module(name="custom-linear", module_class=CustomLinear)
    print(f"Register module response: {register_response}")
    # Check for 500 error
    if register_response.status_code == 500:
        print(f"Error 500: {register_response.text}")
        print(f"Response: {register_response.json() if register_response.text else 'No response body'}")

    print(f"Register module response: {register_response}")
    print(f"After registering module")
    print(client.list_registered_modules())


    # # Test 5: Forward pass
    # print("\n" + "=" * 60)
    # print("Test 5: Forward Pass")
    # print("=" * 60)
    # input_tensor = torch.randn(2, 10)  # batch_size=2, input_dim=10
    # print(f"Input tensor shape: {input_tensor.shape}")
    # print(f"Input tensor:\n{input_tensor}")
    # output = forward(input_tensor)
    # print(f"Output tensor shape: {output.shape}")
    # print(f"Output tensor:\n{output}")
    # print(f"Expected output shape: (2, 5)")
    # print(f"Shape matches: {output.shape == torch.Size([2, 5])}")


    # client = TorchClient(base_url="https://jesterlabs--tinkerbell-torch-training-torchserver-deploy-65957d.modal.run/v1", timeout=10.0)

    # initialize_model(10, 5)

    # # Test 1: Health check
    # print("\n" + "=" * 60)
    # print("Test 1: Health Check")
    # print("=" * 60)
    # health_response = client.health_check()
    # print(f"Health check response: {health_response.json()}")

    # Test 2: Multiply operation
    # print("\n" + "=" * 60)
    # print("Test 2: Multiply Operation")
    # print("=" * 60)
    # data = torch.randn(3, 3)
    # print(f"Original tensor shape: {data.shape}")
    # print(f"Original tensor:\n{data}")
    # value = 2.0
    # print(f"Multiplying by: {value}")
    # result = client.multiply(data, value)
    # result_tensor = deserialize_tensor(result.content)
    # print(f"Result tensor:\n{result_tensor}")
    # print(f"Verification (should be True): {torch.allclose(result_tensor, data * value)}")

    # print("\n" + "=" * 60)
    # print("All tests completed!")
    # print("=" * 60)