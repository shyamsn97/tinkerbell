import httpx
import torch
from transformers import AutoTokenizer
from tinkerbell.types.data import TensorData
WORLD_SIZE = 4
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"
MODEL_NAME = "Qwen/Qwen3-0.6B"

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


def create_training_actors(client: httpx.Client):
    """Create training actors on the server."""
    response = client.post("/create_training_actors", json={
        "model_name": MODEL_NAME,
        "world_size": WORLD_SIZE,
        "master_addr": MASTER_ADDR,
        "master_port": MASTER_PORT,
        "rank": 0,
        "parallelize_plan": parallelize_plan,
    })
    print("Create training actors response:", response.json())
    return response.json()


def check_actor_status(client: httpx.Client):
    """Poll until actors are ready."""
    import time

    while True:
        response = client.post("/get_actor_status", json={
            "model_name": MODEL_NAME,
        })
        status_data = response.json()
        print(f"Actor status: {status_data['status']}")

        if status_data["status"] == "ready":
            print("Actors are ready!")
            break

        time.sleep(2)

def create_inference_actor(client: httpx.Client, model_path: str, tp_size: int, engine_kwargs: dict = {}):
    """Create inference actor on the server."""
    response = client.post("/create_inference_actor", json={
        "model_path": model_path,
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


def forward_backward_example(client: httpx.Client):
    """Tokenize input and perform forward-backward pass."""
    # Load tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Example training batch
    tokenized_inputs = tokenize_input(["The quick brown fox jumps over the lazy dog.", "Machine learning is transforming the world."], tokenizer)

    print(f"Tokenized inputs: {tokenized_inputs['input_ids'].model_dump()}")
    inputs1 = {k:v.slice(0).model_dump() for k,v in tokenized_inputs.items()}
    inputs2 = {k:v.slice(1).model_dump() for k,v in tokenized_inputs.items()}

    print("Zeroing gradients...")
    response = client.post("/zero_grad", json={
        "model_name": MODEL_NAME,
    })
    print("Zero grad response:", response.json())
    print("================================================")
    # Send forward-backward request
    targets1 = inputs1.pop("labels")
    targets2 = inputs2.pop("labels")
    response1 = client.post("/forward_backward", json={
        "model_name": MODEL_NAME,
        "inputs": inputs1,
        "targets": targets1,
        "forward_kwargs": {},
    })
    
    response2 = client.post("/forward_backward", json={
        "model_name": MODEL_NAME,
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
        "model_name": MODEL_NAME,
        "optimizer_params": {},
    })
    print("Optim step response:", response.json())
    print("================================================")

    response1 = client.post("/forward_backward", json={
        "model_name": MODEL_NAME,
        "inputs": inputs1,
        "targets": targets1,
        "forward_kwargs": {},
    })

    response2 = client.post("/forward_backward", json={
        "model_name": MODEL_NAME,
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
    client = httpx.Client(
        base_url="https://jesterlabs--training-service.modal.run",
        timeout=600.0
    )

    # Step 1: Create training actors
    print("=" * 60)
    print("Step 1: Creating training actors")
    print("=" * 60)
    create_training_actors(client)

    # Step 2: Wait for actors to be ready
    print("\n" + "=" * 60)
    print("Step 2: Waiting for actors to be ready")
    print("=" * 60)
    check_actor_status(client)

    # Step 3: Run forward-backward pass
    print("\n" + "=" * 60)
    print("Step 3: Running forward-backward pass")
    print("=" * 60)
    forward_backward_example(client)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)