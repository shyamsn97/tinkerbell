"""
Test client for Tinkerbell training and inference with LoRA.

This script demonstrates:
1. Deploying a Modal server with tensor parallelism
2. Creating training actors with LoRA (Low-Rank Adaptation) configuration
3. Using Renderer to create properly formatted training data with label shifting
4. Training with LoRA adapters for parameter-efficient fine-tuning
5. Saving LoRA-adapted weights and transitioning to inference
6. Running multithreaded and sequential inference tests

LoRA significantly reduces memory usage and training time by only training
low-rank adapter matrices while keeping the base model weights frozen.

The Renderer handles proper label masking and shifting for next-token prediction.
"""

from tinkerbell.client import TrainingClient, ServiceClient
from tinkerbell.types import ModalDeployConfig, LoraConfig
from tinkerbell.renderer import Renderer, TrainOnWhat
from tqdm import tqdm
import time
import asyncio
import concurrent.futures
from functools import partial
import httpx

# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME = "Qwen/Qwen3-0.6B"
GPU_TYPE = "H100"

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE, 
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
    model_id=MODEL_NAME,
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
conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "The answer to 2 + 2 is 4."},
    ],
]

print(f"\nTraining on {len(conversations)} conversations...")

# Use Renderer to create properly formatted training data with masking and shifting
tokenizer = training_client.get_tokenizer()
renderer = Renderer(tokenizer)

# Build training data - train only on assistant responses with proper next-token prediction
training_data = renderer.build_chat_examples(
    conversations=conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

print(f"Created {len(training_data)} training examples with proper label shifting")

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

# Prepare inference prompts (just the user/system messages, not the assistant response)
inference_prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
]

# Tokenize prompts for inference
formatted_prompts = tokenizer.apply_chat_template(inference_prompts, add_generation_prompt=True, tokenize=False)
encoded = tokenizer(formatted_prompts, padding=True, return_tensors="pt")

print("=" * 70)
print("Multithreaded request to generate...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # Submit sampling requests and get TinkerbellFuture objects
    tinkerbell_futures = [sampling_client.sample(input_ids=encoded["input_ids"][i], sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(inference_prompts))]
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
for i in tqdm(range(len(inference_prompts)), desc="Sequential requests"):
    tinkerbell_future = sampling_client.sample(
        input_ids=encoded["input_ids"][i],
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
    tinkerbell_futures = [sampling_client.sample(input_ids=encoded["input_ids"][i], sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(inference_prompts))]
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
for i in tqdm(range(len(inference_prompts)), desc="Sequential requests"):
    tinkerbell_future = sampling_client.sample(
        input_ids=encoded["input_ids"][i],
        sampling_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )
    outputs = tinkerbell_future.result()
print("Last outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)

print("Async request to generate with asyncio.gather (two-phase)...")


async def async_sample_two_phase():
    """
    Two-phase async sampling for maximum efficiency:
    Phase 1: Submit all requests in parallel (first await)
    Phase 2: Poll all results in parallel (second await)
    """
    start_time = time.time()
    
    # Phase 1: Submit all requests and get futures
    print(f"Phase 1: Submitting {len(inference_prompts)} async requests...")
    submit_start = time.time()
    
    async def submit_request(i):
        """First await: submit request and return the future."""
        return await sampling_client.sample_async(
            input_ids=encoded["input_ids"][i],
            sampling_params={"max_new_tokens": 100, "temperature": 0.7}
        )
    
    # Gather all futures (first await for each)
    futures = await asyncio.gather(*[submit_request(i) for i in range(len(inference_prompts))])
    print(f"  ✓ All requests submitted in {time.time() - submit_start:.2f}s")
    
    # Phase 2: Poll all futures for results
    print(f"Phase 2: Polling {len(futures)} futures for results...")
    poll_start = time.time()
    
    # Gather all results (second await for each)
    outputs_list = await asyncio.gather(*futures)
    print(f"  ✓ All results received in {time.time() - poll_start:.2f}s")
    
    elapsed = time.time() - start_time
    print(f"All {len(outputs_list)} async requests completed!")
    print(f"Last output: {outputs_list[-1]}")
    print(f"Total time: {elapsed:.2f} seconds")
    print("=" * 70)
    return outputs_list


# Run the async function
asyncio.run(async_sample_two_phase())
