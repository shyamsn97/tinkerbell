"""
Test client for Tinkerbell training and inference with LoRA.

This script demonstrates:
1. Deploying a Modal server with tensor parallelism
2. Creating training actors with LoRA (Low-Rank Adaptation) configuration
3. Training with LoRA adapters for parameter-efficient fine-tuning
4. Saving LoRA-adapted weights and transitioning to inference
5. Running multithreaded and sequential inference tests

LoRA significantly reduces memory usage and training time by only training
low-rank adapter matrices while keeping the base model weights frozen.
"""

from tinkerbell.client import TrainingClient, ServiceClient
from tinkerbell.types import ModalDeployConfig, LoraConfig
from tinkerbell.types.datum import Datum
from tinkerbell.types.model_input import ModelInput
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial
import httpx

deploy_config = ModalDeployConfig(
    gpu="A100",
    num_gpus=4,
    timeout=86400,
    container_idle_timeout=600,
    max_inputs=200,
    max_wait_time=1200.0,
)

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

# Create training client and train
server_url = "https://jesterlabs--training-service.modal.run"
service_client = ServiceClient(server_url=server_url, timeout=600.0)
print("Service client initialized")
print("Deploying server...")
server_url = service_client.deploy(deploy_config, redeploy=True, wait_for_ready=True)
print("Deployed to: ", server_url)

print("Server health check:")
health_response = service_client.get_health()
print("Health Response: ", health_response)

print("Get Ray Actors:")
ray_actors_response = service_client.get_ray_actors()
ray_actors = ray_actors_response.result()
print("Ray actors: ", ray_actors)

print(f"\nRay Actors ({len(ray_actors)} total):")
for actor_name in ray_actors:
    print(f"  - {actor_name}")

print("Get Store Keys:")
store_keys_response = service_client.get_store_keys()
print("Store keys response: ", store_keys_response)


# client = httpx.Client(base_url=server_url, timeout=60.0)

# print("Health check:")
# health_response = client.get("/health")
# health_response.raise_for_status()
# health_data = health_response.json()
# print("Health: ", health_data)

# # Get store keys
# store_keys_response = client.get("/get_store_keys")
# store_keys_response.raise_for_status()
# print("Store keys response: ", store_keys_response)
# store_keys_data = store_keys_response.json()
# store_keys = store_keys_data.get("keys", [])
# print(f"\nStore keys ({len(store_keys)} total):")
# for key in store_keys:
#     print(f"  - {key}")

# # Get ray actors result
# # List all Ray actors using the server API
# print("\nGetting Ray actors and store keys...")
# ray_actors_response = client.post("/get_ray_actors", content=b"")
# ray_actors_response.raise_for_status()
# ray_actors_data = ray_actors_response.json()
# print("Ray actors data: ", ray_actors_data)

# ray_actors_future = client.post("/get_result", json={"request_id": ray_actors_data["request_id"]})
# ray_actors_future.raise_for_status()
# ray_actors_future = ray_actors_future.json()
# ray_actors = ray_actors_future.get("actor_names", [])
# # ray_actors_response = service_client.get_ray_actors()
# print(f"\nRay Actors ({len(ray_actors)} total):")
# for actor_name in ray_actors:
#     print(f"  - {actor_name}")
# print()
# ray_actors = ray_actors_response.result()
# print(f"\nRay Actors ({len(ray_actors)} total):")
# for actor_name in ray_actors:
#     print(f"  - {actor_name}")
# print()

# ============================================================================
# LoRA CONFIGURATION
# ============================================================================
print("\nConfiguring LoRA (Low-Rank Adaptation) for efficient fine-tuning...")
lora_config = LoraConfig(
    rank=8,  # LoRA rank - higher = more parameters, better quality
    seed=42,  # For reproducible initialization
    train_unembed=False,  # Apply LoRA to the output embedding layer
    train_mlp=True,  # Apply LoRA to MLP/FFN layers
    train_attn=True,  # Apply LoRA to attention layers (Q, K, V, O projections)
)

print(f"LoRA Config: rank={lora_config.rank}, "
      f"train_attn={lora_config.train_attn}, "
      f"train_mlp={lora_config.train_mlp}, "
      f"train_unembed={lora_config.train_unembed}")

training_client = service_client.create_training_client(
    model_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=False,
    lora_config=lora_config.model_dump(),  # Enable LoRA training (convert to dict)
    model_kwargs={
        "torch_dtype": "bfloat16",  # Load in bfloat16 to reduce memory usage (~50% memory savings)
        # "attn_implementation": "flash_attention_2",  # Use Flash Attention 2 for additional speed/memory efficiency
    }
)
training_client.wait_until_ready()

print("Training client ready")

# ============================================================================
# TRAINING EXAMPLE: Forward-Backward Pass
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING EXAMPLE: Running forward-backward passes")
print("=" * 70)

# Create training examples
messages = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the largest planet in our solar system?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many continents are there?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the chemical symbol for gold?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "In what year did World War II end?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the speed of light?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who painted the Mona Lisa?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the smallest unit of matter?"},
    ],
]

formatted_messages = training_client.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

print(f"\nTraining on {len(formatted_messages)} examples...")
print("Texts:")
for i, text in enumerate(formatted_messages):
    print(f"  {i+1}. {text}")

# Tokenize the training examples
tokenizer = training_client.tokenizer

encoded = tokenizer(
    formatted_messages,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Create Datum objects for each training example
training_data = [
    Datum(
        model_input=ModelInput(
            tokens=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            labels=encoded["input_ids"],
        )
    )
]

# Training loop: Run multiple iterations with gradient descent using LoRA!
# Note: Only LoRA adapter parameters are trained, base model weights are frozen
num_training_steps = 5
print(f"\nRunning {num_training_steps} training steps with LoRA adapters...")

bar = tqdm(range(num_training_steps), desc="Training steps")
for step in bar:

    # Zero gradients (only for LoRA parameters)
    training_client.zero_grad()

    print("Forward-backward pass...")
    # Forward-backward pass (only LoRA parameters will accumulate gradients!)
    response = training_client.forward_backward(
        data=training_data,
        forward_kwargs={},
    )
    result = response.result()
    print("Result: ", result)
    print("Forward-backward pass complete")

    losses = result.loss

    if losses:
        avg_loss = sum(losses) / len(losses)
        bar.set_description(f"LoRA Training - Step {step+1}/{num_training_steps} - Loss: {avg_loss:.4f}")

    # Optimizer step with AdamW (only updates LoRA adapter parameters!)
    training_client.optim_step(
        optimizer_params={
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    ).result()

print("\n" + "=" * 70)
print("LoRA Training complete! Loss should have decreased over iterations.")
print("Only LoRA adapter parameters were trained (base model frozen).")
print("=" * 70 + "\n")

# After training, save weights and get inference client
print("Saving LoRA-adapted model weights...")
# Note: LoRA support is automatically enabled because training was done with LoRA
sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/saved-qwen",
    tp_size=1,
    wait_until_ready=True
)
print("Sampling client ready")

# ============================================================================
# INFERENCE EXAMPLE: Text Generation
# ============================================================================
print("\n" + "=" * 70)
print("INFERENCE EXAMPLE: Generating text with trained model")
print("=" * 70 + "\n")

print("=" * 70)
print("Multithreaded request to generate...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # Submit sampling requests and get TinkerbellFuture objects
    tinkerbell_futures = [sampling_client.sample(input_ids=encoded["input_ids"].slice(i), sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    # Now resolve the futures in parallel
    thread_futures = [executor.submit(lambda f: f.result(), tf) for tf in tinkerbell_futures]
    outputs_list = []
    for future in tqdm(concurrent.futures.as_completed(thread_futures), total=len(thread_futures), desc="Multithreaded requests"):
        outputs = future.result()
        outputs_list.append(outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print(f"Generated {len(outputs_list)} outputs")

print("=" * 70)
print("Sequential request to generate...")
start_time = time.time()
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    tinkerbell_future = sampling_client.sample(
        input_ids=encoded["input_ids"].slice(i),
        sampling_params={
            "max_new_tokens": 512,
            "temperature": 0.7,
        },
    )
    outputs = tinkerbell_future.result()
print(f"Time taken: {time.time() - start_time} seconds")

print("Multithreaded request to generate run # 2...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # Submit sampling requests and get TinkerbellFuture objects
    tinkerbell_futures = [sampling_client.sample(input_ids=encoded["input_ids"].slice(i), sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    # Now resolve the futures in parallel
    thread_futures = [executor.submit(lambda f: f.result(), tf) for tf in tinkerbell_futures]
    outputs_list = []
    for future in tqdm(concurrent.futures.as_completed(thread_futures), total=len(thread_futures), desc="Multithreaded requests"):
        outputs = future.result()
        outputs_list.append(outputs)
print("Last outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)

print("Sequential request to generate run # 2...")
start_time = time.time()
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    tinkerbell_future = sampling_client.sample(
        input_ids=encoded["input_ids"].slice(i),
        sampling_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )
    outputs = tinkerbell_future.result()
print("Last outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)
