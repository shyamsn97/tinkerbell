"""
Example demonstrating the future-based API in Tinkerbell.

Most client methods return TinkerbellFuture objects that allow for non-blocking API calls.
Some simple operations (like get_actor_status) execute synchronously and return results directly.

TinkerbellFuture: For async operations that poll the server (e.g., forward_backward)
The request is sent immediately and you can poll for the result using .result()
"""

from tinkerbell.client.service import ServiceClient
from tinkerbell.types.datum import Datum

# Create a service client and training client
service_client = ServiceClient(server_url="http://localhost:8000")

# Create training actors (this returns immediately with a TrainingClient)
training_client = service_client.create_training_client(
    model_id="meta-llama/Llama-3.2-1B",
    tp_size=1,
    wait_until_ready=True,
)

print("=" * 80)
print("Example 1: Basic future usage with forward_backward")
print("=" * 80)

# forward_backward returns a TinkerbellFuture that polls the server
# The request is sent immediately, and you get back a future with a request_id
future = training_client.forward_backward(
    data=[
        Datum(
            model_inputs={"input_ids": [[1, 2, 3]]},
            loss_fn_inputs={"labels": [[1, 2, 3]]},
        )
    ]
)
print(f"Got future with request_id: {future.request_id}")
print(f"Future is done? {future.done()}")

# Do other work here while the server processes the request...
print("Doing other work while server processes...")

# Block and get the actual result by polling the server
result = future.result()
print(f"Result received! Loss: {result.loss}")
print(f"Future is done? {future.done()}")

print("\n" + "=" * 80)
print("Example 2: Multiple concurrent operations")
print("=" * 80)

# Start multiple operations - all requests are sent to the server immediately
futures = []
for i in range(3):
    future = training_client.forward_backward(
        data=[
            Datum(
                model_inputs={"input_ids": [[1 + i, 2 + i, 3 + i]]},
                loss_fn_inputs={"labels": [[1 + i, 2 + i, 3 + i]]},
            )
        ]
    )
    futures.append(future)
    print(f"Started operation {i+1} with request_id: {future.request_id}")

# Now poll and wait for all results
print("\nWaiting for all results...")
results = [f.result() for f in futures]
print(f"All results received! Losses: {[r.loss for r in results]}")

print("\n" + "=" * 80)
print("Example 3: Synchronous operations")
print("=" * 80)

# Some operations like get_actor_status are synchronous and return results directly
# (no future object, executes immediately)
print("Calling get_actor_status() - executes synchronously")
status = training_client.get_actor_status()
print(f"Status: {status.status}")
print("Result received immediately (no polling needed)")

print("\n" + "=" * 80)
print("Example 4: Timeout handling")
print("=" * 80)

try:
    future = training_client.forward_backward(
        data=[
            Datum(
                model_inputs={"input_ids": [[1, 2, 3]]},
                loss_fn_inputs={"labels": [[1, 2, 3]]},
            )
        ]
    )
    print(f"Polling with 0.001s timeout (will likely timeout)...")
    # Set a very short timeout - will likely timeout
    result = future.result(timeout=0.001)
    print(f"Result: {result}")
except TimeoutError as e:
    print(f"✓ Caught timeout error as expected: {e}")

print("\n" + "=" * 80)
print("Example 5: Training loop with futures")
print("=" * 80)

print("Training loop with async forward_backward:")
for step in range(3):
    # Zero gradients (synchronous)
    training_client.zero_grad().result()
    
    # Forward-backward pass (async with polling)
    fb_future = training_client.forward_backward(
        data=[
            Datum(
                model_inputs={"input_ids": [[1, 2, 3]]},
                loss_fn_inputs={"labels": [[1, 2, 3]]},
            )
        ]
    )
    
    # We could do other work here while server processes...
    
    # Get the result
    fb_result = fb_future.result()
    print(f"Step {step+1}: Loss = {fb_result.loss}")
    
    # Optimizer step (synchronous)
    training_client.optim_step().result()

print("\n" + "=" * 80)
print("Example 6: Checking if future is complete")
print("=" * 80)

# Start an operation
future = training_client.forward_backward(
    data=[
        Datum(
            model_inputs={"input_ids": [[1, 2, 3]]},
            loss_fn_inputs={"labels": [[1, 2, 3]]},
        )
    ]
)

# Check if done (non-blocking)
print(f"Is done? {future.done()}")  # False

# Get result (blocking)
result = future.result()

# Check again after result
print(f"Is done after .result()? {future.done()}")  # True

# Calling .result() again returns cached result immediately
result2 = future.result()
print(f"Got cached result: {result2 == result}")  # True

print("\n✅ All examples completed successfully!")
print("\nKey Takeaways:")
print("1. TinkerbellFuture: Sends request immediately, polls server on .result()")
print("2. Futures provide .done() to check completion and .result() to get values")
print("3. Can batch multiple operations before calling .result() for concurrency")
print("4. Some simple operations (like get_actor_status) are synchronous")
