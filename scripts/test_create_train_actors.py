import httpx
import torch
from transformers import AutoTokenizer
from tinkerbell.types.data import TensorData

WORLD_SIZE = 2
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME_2 = "Qwen/Qwen2-0.5B-Instruct"

# Define parallelization plan
parallelize_plan = {
    # Attention projections (all layers)
    "model.layers.*.self_attn.q_proj": "column",
    "model.layers.*.self_attn.k_proj": "column",
    "model.layers.*.self_attn.v_proj": "column",
    "model.layers.*.self_attn.o_proj": "row",

    # MLP projections (all layers)
    "model.layers.*.mlp.gate_proj": "column",
    "model.layers.*.mlp.up_proj": "column",
    "model.layers.*.mlp.down_proj": "row",
}


def create_training_actors(client: httpx.Client, model_id: str):
    """Create training actors on the server."""
    response = client.post("/create_training_actors", json={
        "model_id": model_id,
        "world_size": WORLD_SIZE,
        "parallelize_plan": parallelize_plan,
    })
    print("Create training actors response:", response.json())
    return response.json()


def check_actor_status(client: httpx.Client, model_id: str):
    """Poll until actors are ready."""
    import time

    while True:
        response = client.post("/get_actor_status", json={
            "model_id": model_id,
        })
        status_data = response.json()
        print(f"Actor status: {status_data['status']}")

        if status_data["status"] == "ready":
            print("Actors are ready!")
            break

        time.sleep(2)

def create_inference_actor(client: httpx.Client, model_id: str, tp_size: int, engine_kwargs: dict = {}):
    """Create inference actor on the server."""
    response = client.post("/create_inference_actor", json={
        "model_id": model_id,
        "tp_size": tp_size,
        "engine_kwargs": engine_kwargs,
    })
    print("Create inference actor response:", response.json())
    return response.json()

def tokenize_input(texts: list[str], tokenizer: AutoTokenizer) -> dict[str, list[int]]:
    """Tokenize input texts."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {
        "input_ids": TensorData.from_torch(encoded["input_ids"]),  # List[List[int]]
        "attention_mask": TensorData.from_torch(encoded["attention_mask"]),  # List[List[int]]
        "labels": TensorData.from_torch(encoded["input_ids"]),  # List[List[int]]
    }
    return inputs


def forward_backward_example(client: httpx.Client, model_id: str):
    """Tokenize input and perform forward-backward pass."""
    # Load tokenizer
    print(f"\nLoading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Example training batch
    tokenized_inputs = tokenize_input(["The quick brown fox jumps over the lazy dog.", "Machine learning is transforming the world."], tokenizer)

    print(f"Tokenized inputs: {tokenized_inputs['input_ids'].model_dump()}")
    inputs1 = {k:v.slice(0).model_dump() for k,v in tokenized_inputs.items()}
    inputs2 = {k:v.slice(1).model_dump() for k,v in tokenized_inputs.items()}

    print("Zeroing gradients...")
    response = client.post("/zero_grad", json={
        "model_id": model_id,
    })
    print("Zero grad response:", response.json())
    print("================================================")
    # Send forward-backward request
    targets1 = inputs1.pop("labels")
    targets2 = inputs2.pop("labels")
    response1 = client.post("/forward_backward", json={
        "model_id": model_id,
        "inputs": inputs1,
        "targets": targets1,
        "forward_kwargs": {},
    })
    
    response2 = client.post("/forward_backward", json={
        "model_id": model_id,
        "inputs": inputs2,
        "targets": targets2,
        "forward_kwargs": {},
    })

    result1 = response1.json()
    result2 = response2.json()
    print("Request 1", result1)
    print("Request 2", result2)
    output1 = client.post("/get_result", json={
        "request_id":result1['request_id']
    })
    print("Output 1", output1)
    print(f"Loss: {output1.json()['loss']}")
    output2 = client.post("/get_result", json={
        "request_id":result2['request_id']
    })
    print("Output 2", output2)
    print(f"Loss: {output2.json()['loss']}")
    print("Backward pass complete!")
    print("================================================")

    print("Optimizing...")
    response = client.post("/optim_step", json={
        "model_id": model_id,
        "optimizer_params": {},
    })
    print("Optim step response:", response.json())
    print("================================================")

    response1 = client.post("/forward_backward", json={
        "model_id": model_id,
        "inputs": inputs1,
        "targets": targets1,
        "forward_kwargs": {},
    })

    response2 = client.post("/forward_backward", json={
        "model_id": model_id,
        "inputs": inputs2,
        "targets": targets2,
        "forward_kwargs": {},
    })

    result1 = response1.json()
    result2 = response2.json()
    print("Request 1 after optimization", result1)
    print("Request 2 after optimization", result2)
    output1 = client.post("/get_result", json={
        "request_id":result1['request_id']
    })
    print("Output 1 after optimization", output1)
    print(f"Loss: {output1.json()['loss']}")
    output2 = client.post("/get_result", json={
        "request_id":result2['request_id']
    })
    print("Output 2 after optimization", output2)
    print(f"Loss: {output2.json()['loss']}")
    print("Backward pass complete!")
    print("================================================")

    return output1, output2

if __name__ == "__main__":
    import sys
    
    client = httpx.Client(
        base_url="https://jesterlabs--training-service.modal.run",
        timeout=600.0
    )
    
    # Check if user wants to test inference
    if len(sys.argv) > 1 and sys.argv[1] == "test_inference":
        # Example: python test_create_train_actors.py test_inference meta-llama/Llama-3.1-8B
        inference_model = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.1-8B"
        num_threads = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        print("=" * 70)
        print("INFERENCE BATCHING TEST")
        print("=" * 70)
        print(f"Model: {inference_model}")
        print(f"Threads: {num_threads}")
        print("=" * 70)
        
        # Create inference actor if needed
        print("\nCreating/checking inference actor...")
        create_inference_actor(client, inference_model, tp_size=1)
        check_inference_actor_status(client, inference_model)
        
        # Run multithreaded test
        results = test_mutlithreaded_inference(client, inference_model, num_threads)
        
        print("\n" + "=" * 70)
        print("Test Complete!")
        print("=" * 70)
        sys.exit(0)

    # Step 1: Create training actors
    print("=" * 60)
    print("Step 1: Creating training actors")
    print("=" * 60)
    create_training_actors(client, MODEL_NAME)
    create_training_actors(client, MODEL_NAME_2)

    # Step 2: Wait for actors to be ready
    print("\n" + "=" * 60)
    print("Step 2: Waiting for actors to be ready")
    print("=" * 60)
    check_actor_status(client, MODEL_NAME)
    check_actor_status(client, MODEL_NAME_2)

    # Step 3: Run forward-backward pass
    print("\n" + "=" * 60)
    print("Step 3: Running forward-backward pass")
    print("=" * 60)
    print(f"Running forward-backward pass for {MODEL_NAME}")
    print("=" * 60)
    forward_backward_example(client, MODEL_NAME)
    print("=" * 60)
    print(f"Running forward-backward pass for {MODEL_NAME_2}")
    print("=" * 60)
    forward_backward_example(client, MODEL_NAME_2)
    print("=" * 60)
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def call_generate(client: httpx.Client, model_id: str, thread_id: int, results: list):
    """Single threaded call to generate endpoint."""
    import time
    start = time.time()
    
    try:
        response = client.post("/generate", json={
            "model_id": model_id,
            "prompts": [f"Thread {thread_id}: Tell me a very short story about AI"],
            "sampling_params": {"max_new_tokens": 50, "temperature": 0.7}
        })
        response.raise_for_status()
        elapsed = time.time() - start
        
        result = response.json()
        print(f"✓ Thread {thread_id} completed in {elapsed:.2f}s")
        print(f"  Output preview: {result['outputs'][0][:80]}...")
        results.append((elapsed, result))
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Thread {thread_id} failed after {elapsed:.2f}s: {e}")
        results.append((elapsed, None))


def test_mutlithreaded_inference(client: httpx.Client, model_id: str, num_threads: int = 10):
    """Test multithreaded inference to verify concurrent batching with SGLang.
    
    This test sends multiple concurrent requests to verify that:
    1. FastAPI handles concurrent HTTP requests
    2. Ray actor processes them concurrently (with max_concurrency)
    3. SGLang's continuous batching kicks in to process them efficiently
    """
    import threading
    import time
    
    print("\n" + "=" * 70)
    print(f"Testing Multithreaded Inference with {num_threads} concurrent threads")
    print("=" * 70)
    
    results = []
    threads = []
    start_time = time.time()
    
    # Launch all threads
    print(f"Launching {num_threads} concurrent requests...")
    for i in range(num_threads):
        thread = threading.Thread(
            target=call_generate,
            args=(client, model_id, i, results)
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = len([r for r in results if r[1] is not None])
    if successful > 0:
        avg_latency = sum(r[0] for r in results if r[1] is not None) / successful
    else:
        avg_latency = 0
    
    print("\n" + "=" * 70)
    print("Multithreaded Inference Results:")
    print("=" * 70)
    print(f"Total wall-clock time: {total_time:.2f}s")
    print(f"Successful requests: {successful}/{num_threads}")
    print(f"Average latency per request: {avg_latency:.2f}s")
    print(f"Throughput: {successful/total_time:.2f} requests/second")
    
    # Determine if batching is working
    print("\n" + "=" * 70)
    print("Batching Analysis:")
    print("=" * 70)
    
    # If requests were sequential, total time ≈ avg_latency * num_threads
    # If batched, total time ≈ avg_latency (all complete together)
    expected_sequential_time = avg_latency * num_threads
    
    if total_time < expected_sequential_time * 0.5:
        print("✅ BATCHING IS WORKING!")
        print(f"   → Concurrent execution: {total_time:.2f}s")
        print(f"   → Sequential would take: ~{expected_sequential_time:.2f}s")
        print(f"   → Speedup: {expected_sequential_time/total_time:.1f}x")
        print(f"   → SGLang is successfully batching concurrent requests!")
    else:
        print("⚠️  BATCHING MAY NOT BE WORKING!")
        print(f"   → Expected concurrent time: ~{avg_latency:.2f}s")
        print(f"   → Actual time: {total_time:.2f}s")
        print(f"   → Requests appear to be processed sequentially")
        print(f"")
        print(f"   Possible issues:")
        print(f"   1. max_concurrency not set on Ray actor")
        print(f"   2. Server not using async properly")
        print(f"   3. Network/infrastructure bottlenecks")
    print("=" * 70)
    
    return results


def check_inference_actor_status(client: httpx.Client, model_id: str):
    """Poll until inference actor is ready."""
    import time
    
    print(f"\nWaiting for inference actor for {model_id} to be ready...")
    while True:
        response = client.post("/get_inference_actor_status", json={
            "model_id": model_id,
        })
        status_data = response.json()
        print(f"Inference actor status: {status_data['status']}")
        
        if status_data["status"] == "ready":
            print("✓ Inference actor is ready!")
            break
        
        time.sleep(2)

