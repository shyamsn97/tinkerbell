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
service_client = ServiceClient(timeout=600.0)
server_url = service_client.deploy(deploy_config)
print("Deployed to: ", server_url)

# List all Ray actors using the server API
print("\nRay actors:")
ray_actors_response = service_client.get_ray_actors()
for actor_name in ray_actors_response.actor_names:
    print(f"  - {actor_name}")
print()

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
    model_id="Qwen/Qwen3-0.6B",
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=False,
    lora_config=lora_config.model_dump(),  # Enable LoRA training (convert to dict)
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
training_data = []
for i in range(len(formatted_messages)):
    datum = Datum(
        model_input=ModelInput(
            tokens=encoded["input_ids"].slice(i),
            attention_mask=encoded["attention_mask"].slice(i),
        ),
        loss_fn_inputs={
            "labels": encoded["input_ids"].slice(i),  # Use input_ids as labels for language modeling
        }
    )
    training_data.append(datum)

# Training loop: Run multiple iterations with gradient descent using LoRA!
# Note: Only LoRA adapter parameters are trained, base model weights are frozen
num_training_steps = 5
print(f"\nRunning {num_training_steps} training steps with LoRA adapters...")

bar = tqdm(range(num_training_steps), desc="Training steps")
for step in bar:

    # Zero gradients (only for LoRA parameters)
    training_client.zero_grad()

    # Forward-backward pass (only LoRA parameters will accumulate gradients!)
    response = training_client.forward_backward(
        data=training_data,
        forward_kwargs={},
    )

    # Get the result
    result = training_client.get_result(response.request_id)
    losses = result.get("loss", [])

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
    )

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
    futures = [executor.submit(sampling_client.sample, input_ids=encoded["input_ids"].slice(i), sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multithreaded requests"):
        outputs = future.result()
print(f"Time taken: {time.time() - start_time} seconds")

print("=" * 70)
print("Sequential request to generate...")
start_time = time.time()
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    outputs = sampling_client.sample(
        input_ids=encoded["input_ids"].slice(i),
        sampling_params={
            "max_new_tokens": 512,
            "temperature": 0.7,
        },
    )
print(f"Time taken: {time.time() - start_time} seconds")

print("Multithreaded request to generate run # 2...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(sampling_client.sample, input_ids=encoded["input_ids"].slice(i), sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multithreaded requests"):
        outputs = future.result()
print("Outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)

print("Sequential request to generate run # 2...")
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    outputs = sampling_client.sample(
        input_ids=encoded["input_ids"].slice(i),
        sampling_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )
print("Outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)
