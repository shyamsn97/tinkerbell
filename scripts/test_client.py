from pydoc import text
from tinkerbell.client import TrainingClient, ServiceClient
from tinkerbell.types import ModalDeployConfig
from tinkerbell.types.datum import Datum
from tinkerbell.types.model_input import ModelInput

import time
import concurrent.futures
from functools import partial

from tqdm import tqdm

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


training_client = service_client.create_training_client(
    model_id="Qwen/Qwen3-0.6B",
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=True,
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
training_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world.",
    "Python is a popular programming language.",
    "Neural networks can learn complex patterns.",
]

print(f"\nTraining on {len(training_texts)} examples...")
print("Texts:")
for i, text in enumerate(training_texts):
    print(f"  {i+1}. {text}")

# Tokenize the training examples
tokenizer = training_client.tokenizer
encoded = tokenizer(
    training_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Create Datum objects for each training example
training_data = []
for i in range(len(training_texts)):
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

# Training loop: Run multiple iterations with gradient descent
num_training_steps = 5
print(f"\nRunning {num_training_steps} training steps...")

for step in range(num_training_steps):
    print(f"\n--- Training Step {step + 1}/{num_training_steps} ---")
    
    # Zero gradients
    training_client.zero_grad()
    
    # Forward-backward pass
    response = training_client.forward_backward(
        data=training_data,
        forward_kwargs={},
    )
    
    # Get the result
    result = training_client.get_result(response.request_id)
    losses = result.get("loss", [])
    
    if losses:
        avg_loss = sum(losses) / len(losses)
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Individual Losses: {[f'{loss:.4f}' for loss in losses]}")
    
    # Optimizer step with AdamW
    training_client.optim_step(
        optimizer_params={
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    )
    print(f"  ✓ Gradients applied")

print("\n" + "=" * 70)
print("Training complete! Loss should have decreased over iterations.")
print("=" * 70 + "\n")

# After training, save weights and get inference client
print("Saving trained model weights...")
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

messages = [
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the largest planet in our solar system?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "How many continents are there?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the chemical symbol for gold?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "In what year did World War II end?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the speed of light?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "Who painted the Mona Lisa?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the smallest unit of matter?"},
        {"role": "system", "content": "<think>"},
    ],
]

formatted_messages = training_client.tokenizer.apply_chat_template(messages)

print("=" * 70)
print("Multithreaded request to generate...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(sampling_client.generate, text=[formatted_messages[i]], sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multithreaded requests"):
        outputs = future.result()
print(f"Time taken: {time.time() - start_time} seconds")

print("=" * 70)
print("Sequential request to generate...")
start_time = time.time()
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    outputs = sampling_client.generate(
        text=[formatted_messages[i]],
        sampling_params={
            "max_new_tokens": 512,
            "temperature": 0.7,
        },
    )
print(f"Time taken: {time.time() - start_time} seconds")

print("Multithreaded request to generate run # 2...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(sampling_client.generate, text=[formatted_messages[i]], sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multithreaded requests"):
        outputs = future.result()
print("Outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)

print("Sequential request to generate run # 2...")
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    outputs = sampling_client.generate(
        text=[formatted_messages[i]],
        sampling_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )
print("Outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)
