"""
Simple example showing the exact usage:

All operations now send requests immediately and return futures.
Calling .result() polls the server until ready.
"""

from tinkerbell.client.training import TrainingClient
from tinkerbell.types.datum import Datum

# Initialize training client
training_client = TrainingClient(
    server_url="http://localhost:8000",
    model_id="meta-llama/Llama-3.2-1B",
)

print("=" * 80)
print("Example 1: forward_backward - TRUE ASYNC")
print("=" * 80)

# This sends the request to the server IMMEDIATELY
# Server returns request_id and processes in background
output = training_client.forward_backward(
    data=[
        Datum(
            model_inputs={"input_ids": [[1, 2, 3, 4, 5]]},
            loss_fn_inputs={"labels": [[1, 2, 3, 4, 5]]},
        )
    ]
)

print(f"✅ Request sent to server with ID: {output.request_id}")
print(f"✅ Server is processing in background...")
print(f"✅ Future done? {output.done()}")

# You can do other work here while the server processes...
print("Doing other work...")

# This creates a light httpx client and polls /poll_result
print("Now polling server for result...")
actual_result = output.result()

print(f"✅ Got result! Loss: {actual_result.loss}")
print(f"✅ Future done? {output.done()}")

# You can call .result() again and it returns the cached value
cached_result = output.result()
print(f"✅ Cached result: {cached_result.loss}")

print("\n" + "=" * 80)
print("Example 2: zero_grad - ALSO ASYNC NOW")
print("=" * 80)

# ALL operations are now async - request sent immediately
zero_grad_future = training_client.zero_grad()
print(f"✅ zero_grad request sent with ID: {zero_grad_future.request_id}")
print(f"✅ Future done? {zero_grad_future.done()}")

# Poll for result
result = zero_grad_future.result()
print(f"✅ Got result: {result}")

print("\n" + "=" * 80)
print("Example 3: sample - ASYNC TOO")
print("=" * 80)

from tinkerbell.client.sampling import SamplingClient

sampling_client = SamplingClient(
    server_url="http://localhost:8000",
    model_id="meta-llama/Llama-3.2-1B",
)

# Request sent immediately, server processes in background
sample_future = sampling_client.sample(
    prompts=["Hello, how are you?"],
    sampling_params={"temperature": 0.8, "max_tokens": 100},
)

print(f"✅ Sample request sent with ID: {sample_future.request_id}")

# Do other work while sampling happens...
print("Server is generating text in background...")

# Poll until done
sample_result = sample_future.result()
print(f"✅ Sample result: {sample_result.outputs}")

print("\n" + "=" * 80)
print("Example 4: Multiple concurrent operations")
print("=" * 80)

# Start 3 operations - all sent immediately
futures = [
    training_client.forward_backward(
        data=[Datum(model_inputs={"input_ids": [[i, i+1, i+2]]}, 
                    loss_fn_inputs={"labels": [[i, i+1, i+2]]})]
    )
    for i in range(1, 4)
]

print(f"✅ Started 3 operations:")
for i, f in enumerate(futures, 1):
    print(f"   Operation {i}: request_id={f.request_id}")

print("Server processing all 3 in parallel...")

# Now poll for all results
results = [f.result() for f in futures]
print(f"✅ All results: {[r.loss for r in results]}")

print("\n" + "=" * 80)
print("✅ SUCCESS! The API now works correctly:")
print("=" * 80)
print("1. ✅ ALL operations send request IMMEDIATELY")
print("2. ✅ Server returns request_id and processes in background")
print("3. ✅ .result() creates light httpx client and POLLS")
print("4. ✅ Server stores results in global store")
print("5. ✅ Future polls /poll_result endpoint until ready")
print("6. ✅ NO MORE SYNC FUTURES - everything is async!")
