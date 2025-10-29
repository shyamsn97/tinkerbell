import httpx
import torch
from transformers import AutoTokenizer

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
        "model_kwargs": {},
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


def forward_backward_example(client: httpx.Client):
    """Tokenize input and perform forward-backward pass."""
    # Load tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Example training batch
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
    ]

    # Tokenize with padding and attention mask
    print(f"Tokenizing {len(texts)} examples...")
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Convert tensors to lists for JSON serialization
    # NOTE: Torch tensors are NOT JSON-serializable, so we must convert to lists
    # The server will reconstruct tensors from these lists
    inputs = {
        "input_ids": encoded["input_ids"].tolist(),  # List[List[int]]
        "attention_mask": encoded["attention_mask"].tolist(),  # List[List[int]]
        "labels": encoded["input_ids"].tolist(),  # List[List[int]]
    }

    # You can also send float tensors the same way:
    # Example: Position embeddings or custom weights
    # float_example = torch.randn(2, 4)  # Some float tensor
    # float_list = float_example.tolist()  # Converts to List[List[float]]

    print(f"Input shapes:")
    print(f"  - input_ids: {encoded['input_ids'].shape} (dtype: {encoded['input_ids'].dtype})")
    print(f"  - attention_mask: {encoded['attention_mask'].shape} (dtype: {encoded['attention_mask'].dtype})")
    # print(f"  - float_example: {float_example.shape} (dtype: {float_example.dtype})")
    print(f"\nAll tensors converted to lists for JSON serialization")
    print(f"Sending forward_backward request...")

    # Send forward-backward request
    response = client.post("/forward_backward", json={
        "model_name": MODEL_NAME,
        "inputs": inputs,
        "forward_kwargs": {},
        "model_kwargs": {},
    })

    result = response.json()
    print(f"Loss: {result['loss']}")
    return result


if __name__ == "__main__":
    client = httpx.Client(
        base_url="https://jesterlabs--training-service.modal.run",
        timeout=300.0
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