# """Test script for TorchTrainingServer and TorchTrainingClient."""

# import multiprocessing
# import time
# from typing import Optional

# import torch
# import torch.nn as nn

# from tinkerbell.client.torch_client import TorchTrainingClient
# from tinkerbell.server.training.torch_server import TorchTrainingServer


# class SimpleModel(nn.Module):
#     """Simple neural network for testing."""

#     def __init__(
#         self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5
#     ):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


# def run_server(port: int = 8000):
#     """Run the training server in a separate process.

#     Args:
#         port: Port to run the server on
#     """
#     # Create a simple model
#     model = SimpleModel()

#     # Create and deploy the server
#     server = TorchTrainingServer(model, port=port)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     server.set_optimizer(optimizer)

#     server.deploy()


# def test_client(port: int = 8000, wait_time: int = 2):
#     """Test the client connection to the server.

#     Args:
#         port: Port where the server is running
#         wait_time: Time to wait for server to start (seconds)
#     """
#     # Wait for server to start
#     time.sleep(wait_time)

#     # Create client
#     client = TorchTrainingClient(base_url=f"http://localhost:{port}")

#     print("=" * 60)
#     print("Testing Torch Training Server/Client")
#     print("=" * 60)

#     try:
#         # Test 1: Health check
#         print("\n1. Testing health check...")
#         health = client.health_check()
#         print(f"   ✓ Server is healthy: {health}")

#         # Test 2: Forward pass
#         print("\n2. Testing forward pass...")
#         test_input = torch.randn(4, 10)  # Batch of 4, input size 10
#         output = client.forward([test_input])
#         print(f"   ✓ Forward pass successful")
#         print(f"   Input shape: {test_input.shape}")
#         print(f"   Output shape: {output.shape}")
#         print(f"   Output sample: {output[0, :3]}")

#         # Test 3: Forward pass with loss
#         print("\n3. Testing forward pass with loss...")
#         test_input = torch.randn(4, 10)
#         test_target = torch.randint(0, 5, (4,))  # 4 samples, 5 classes
#         loss_fn = nn.CrossEntropyLoss()
#         loss = client.forward([test_input, test_target], loss_fn)
#         print(f"   ✓ Forward with loss successful")
#         print(f"   Loss value: {loss.item():.4f}")

#         # Test 4: Forward-backward pass
#         print("\n4. Testing forward-backward pass...")
#         loss = client.forward_backward([test_input, test_target], loss_fn)
#         print(f"   ✓ Forward-backward pass successful")
#         print(f"   Loss value: {loss:.4f}")

#         # Test 5: Optimizer step
#         print("\n5. Testing optimizer step...")
#         result = client.optim_step()
#         print(f"   ✓ Optimizer step successful: {result}")

#         # Test 6: Save state
#         print("\n6. Testing save state...")
#         state = client.save_state()
#         print(f"   ✓ State saved successfully")
#         print(f"   State keys: {list(state.keys())}")

#         # Test 7: Load state
#         print("\n7. Testing load state...")
#         result = client.load_state(state)
#         print(f"   ✓ State loaded successfully: {result}")

#         # Test 8: Large tensor transfer
#         print("\n8. Testing large tensor transfer...")
#         large_input = torch.randn(100, 10)  # Larger batch
#         output = client.forward([large_input])
#         print(f"   ✓ Large tensor transfer successful")
#         print(f"   Input shape: {large_input.shape}")
#         print(f"   Output shape: {output.shape}")

#         # Test 9: Benchmark speed
#         print("\n9. Benchmarking transfer speed...")
#         num_iterations = 10
#         start_time = time.time()
#         for _ in range(num_iterations):
#             test_input = torch.randn(4, 10)
#             _ = client.forward([test_input])
#         elapsed = time.time() - start_time
#         avg_time = elapsed / num_iterations
#         print(f"   ✓ Average time per forward pass: {avg_time*1000:.2f} ms")
#         print(f"   Throughput: {num_iterations/elapsed:.2f} requests/sec")

#         print("\n" + "=" * 60)
#         print("All tests passed! ✓")
#         print("=" * 60)

#     except Exception as e:
#         print(f"\n✗ Error: {e}")
#         import traceback

#         traceback.print_exc()
#     finally:
#         client.close()


# def main():
#     """Main function to run the test."""
#     port = 8000

#     print("Starting test...")
#     print(f"Server will run on port {port}")

#     # Start server in a separate process
#     server_process = multiprocessing.Process(target=run_server, args=(port,))
#     server_process.start()

#     try:
#         # Run client tests
#         test_client(port=port)
#     finally:
#         # Cleanup
#         print("\nShutting down server...")
#         server_process.terminate()
#         server_process.join(timeout=5)
#         if server_process.is_alive():
#             server_process.kill()
#         print("Server stopped.")


# if __name__ == "__main__":
#     main()
