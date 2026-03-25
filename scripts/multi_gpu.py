from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig, LoraConfig
from tinkerbell.renderer import Renderer, TrainOnWhat
from tqdm import tqdm
import time


# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME = "Qwen/Qwen3-0.6B"
GPU_TYPE = "A100"

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE, 
    num_gpus=6,
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
service_client = ServiceClient.deploy_or_connect(deploy_config)
print("Service client initialized")
print("Deploying server...")
print("Deployed to: ", service_client.server_url)

print("Server health check:")
health_response = service_client.get_health()
print("Health Response: ", health_response)

print("Get Ray Actors:")
ray_actors = service_client.get_ray_actors()
print("Ray actors: ", ray_actors)

print(f"\nRay Actors ({len(ray_actors)} total):")
for actor_name in ray_actors:
    print(f"  - {actor_name}")

print("Get Store Keys:")
store_keys_response = service_client.get_store_keys()
print("Store keys response: ", store_keys_response)

# ============================================================================
# MULTI-LORA CONFIGURATION - Training with Multiple Adapters
# ============================================================================
print("\n" + "=" * 70)
print("MULTI-LORA SETUP: Creating training clients with multiple adapters")
print("=" * 70)

# Adapter 1: High-rank adapter for attention layers (better quality, more parameters)
print("\n1. Creating first adapter (high-rank, attention-focused)...")
lora_config_1 = LoraConfig(
    rank=8,  # Higher rank = more parameters, better expressiveness
    seed=42,
    train_unembed=False,
    train_mlp=False,  # Only train attention for this adapter
    train_attn=True,
)
print(f"   Adapter 1: rank={lora_config_1.rank}, train_attn={lora_config_1.train_attn}, train_mlp={lora_config_1.train_mlp}")

# model_name defaults to cleaned base_model ("qwen_qwen3-0.6b")
# But for multi-LoRA, we explicitly set model_name so adapters share the same actor group
training_client = service_client.create_training_client(
    base_model=MODEL_NAME,
    tp_size=2,
    model_name="multi_lora_model",  # Explicit name for multi-adapter support
    adapter_name="task1_attention",
    lora_config=lora_config_1.model_dump(),
    parallelize_plan=parallelize_plan,
    model_kwargs={"torch_dtype": "bfloat16"},
)

# Adapter 2: Low-rank adapter for MLP layers (efficient, task-specific)
print("\n2. Creating second adapter on the SAME base model (low-rank, MLP-focused)...")
lora_config_2 = LoraConfig(
    rank=4,  # Lower rank = fewer parameters, more efficient
    seed=42,
    train_unembed=False,
    train_mlp=True,  # Only train MLP for this adapter
    train_attn=False,
)
print(f"   Adapter 2: rank={lora_config_2.rank}, train_attn={lora_config_2.train_attn}, train_mlp={lora_config_2.train_mlp}")

# SAME model_name = shared actor group (multi-LoRA on one base model)
training_client_2 = service_client.create_training_client(
    base_model=MODEL_NAME,
    tp_size=2,
    model_name="multi_lora_model",  # Same name = shared actors!
    adapter_name="task2_mlp",
    lora_config=lora_config_2.model_dump(),
    parallelize_plan=parallelize_plan,
    model_kwargs={"torch_dtype": "bfloat16"},
)

print("\n3. Creating full model training client (no LoRA)...")
# Different model_name = separate actor group
full_model_client = service_client.create_training_client(
    base_model=MODEL_NAME,
    tp_size=2,
    model_name="full_model",  # Different name = separate actors
    parallelize_plan=parallelize_plan,
    model_kwargs={"torch_dtype": "bfloat16"},
)

training_client.wait_until_ready()
print("   ✓ Adapter 'task1_attention' ready")

training_client_2.wait_until_ready()
print("   ✓ Adapter 'task2_mlp' ready")

full_model_client.wait_until_ready()
print("   ✓ Full model client ready")
print("=" * 70)

# ============================================================================
# TRAINING EXAMPLE: Multi-Adapter Training with Separate Clients
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING EXAMPLE: Multi-adapter training with separate clients")
print("Get Ray Actors:")
ray_actors = service_client.get_ray_actors()
print("Ray actors: ", ray_actors)
print("=" * 70)

# Create different training examples for different adapters
# Task 1 (Adapter 1 - Attention): Geography questions
task1_conversations = [
    [
        {"role": "system", "content": "You are a geography expert."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    [
        {"role": "system", "content": "You are a geography expert."},
        {"role": "user", "content": "What is the largest ocean?"},
        {"role": "assistant", "content": "The Pacific Ocean is the largest ocean on Earth."},
    ],
]

# Task 2 (Adapter 2 - MLP): Math questions
task2_conversations = [
    [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "The answer to 2 + 2 is 4."},
    ],
    [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 5 * 7?"},
        {"role": "assistant", "content": "5 multiplied by 7 equals 35."},
    ],
]

# Full model: General knowledge (uses all parameters)
full_model_conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about machine learning."},
        {"role": "assistant", "content": "Machine learning is a branch of AI that enables computers to learn from data."},
    ],
]

print(f"\nTask 1 (Adapter 1): {len(task1_conversations)} geography examples")
print(f"Task 2 (Adapter 2): {len(task2_conversations)} math examples")
print(f"Full model: {len(full_model_conversations)} general examples")

# Use Renderer to create properly formatted training data
tokenizer = training_client.get_tokenizer()
renderer = Renderer(tokenizer)

# Build training data for each task with adapter routing
task1_data = renderer.build_chat_samples(
    messages=task1_conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

task2_data = renderer.build_chat_samples(
    messages=task2_conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

full_model_data = renderer.build_chat_samples(
    messages=full_model_conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

print(f"\nCreated {len(task1_data)} task1 examples, {len(task2_data)} task2 examples, {len(full_model_data)} full model examples")

# Training loop: Train both adapters separately with their own clients!
num_training_steps = 2
print(f"\n{'=' * 70}")
print(f"Training {num_training_steps} steps with separate clients for each adapter")
print(f"{'=' * 70}")

bar = tqdm(range(num_training_steps), desc="Multi-adapter training")
for step in bar:
    # === Adapter 1 Training (task1_attention) ===

    training_client.zero_grad().result()
    training_client_2.zero_grad().result()
    full_model_client.zero_grad().result()
    response = training_client.forward_backward(
        data=task1_data,
        forward_kwargs={},
    )

    response_2 = training_client_2.forward_backward(
        data=task2_data,
        forward_kwargs={},
    )

    full_response = full_model_client.forward_backward(
        data=full_model_data,
        forward_kwargs={},
    )

    result = response.result()
    result_2 = response_2.result()
    full_result = full_response.result()

    losses = result.loss
    losses_2 = result_2.loss
    full_losses = full_result.loss

    if losses:
        avg_loss = sum(losses) / len(losses)
        print(f"  ✓ Adapter 1 Loss: {avg_loss:.4f}")
        bar.set_description(f"Step {step+1} - Adapter 1 Loss: {avg_loss:.4f}")

    if losses_2:
        avg_loss_2 = sum(losses_2) / len(losses_2)
        print(f"  ✓ Adapter 2 Loss: {avg_loss_2:.4f}")

    if full_losses:
        full_avg_loss = sum(full_losses) / len(full_losses)
        print(f"  ✓ Full Model Loss: {full_avg_loss:.4f}")

    print(f"\n--- Step {step+1}: Adapter 1 Training (task1_attention) ---")

    training_client.optim_step(
        optimizer_params={
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    ).result()

    # === Adapter 2 Training (task2_mlp) ===
    print(f"--- Step {step+1}: Adapter 2 Training (task2_mlp) ---")

    training_client_2.optim_step(
        optimizer_params={
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    ).result()
    
    # === Full Model Training ===
    print(f"--- Step {step+1}: Full Model Training ---")

    full_model_client.optim_step(
        optimizer_params={
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    ).result()

print("\n" + "=" * 70)
print("✓ Multi-adapter training complete!")
print("  - Adapter 1 (task1_attention): Trained on geography (separate client)")
print("  - Adapter 2 (task2_mlp): Trained on math (separate client)")
print("  - Full model: Trained on general knowledge")
print("=" * 70 + "\n")

# ============================================================================
# INFERENCE SETUP: Save weights and create sampling clients
# ============================================================================
print("\n" + "=" * 70)
print("INFERENCE SETUP: Saving weights and creating sampling clients")
print("=" * 70)

# Save adapter 1
# NOTE: Training actors are still running, so we reduce mem_fraction_static
# to avoid OOM when allocating SGLang's KV cache
print("\n1. Saving adapter 1 (task1_attention)...")
lora_sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/task1-attention-qwen",
    tp_size=1,
    wait_until_ready=False,
)

# Save adapter 2 - reuses the same sampling actor as adapter 1!
print("\n2. Saving adapter 2 (task2_mlp)...")
lora_sampling_client_2 = training_client_2.save_weights_and_get_sampling_client(
    checkpoint_path="/models/task2-mlp-qwen",
    tp_size=1,
    wait_until_ready=False,
)

# Save full model - creates a separate sampling actor (different model_name)
print("\n3. Saving full model (no LoRA)...")
full_sampling_client = full_model_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/full-qwen",
    tp_size=1,
    wait_until_ready=False,
)

lora_sampling_client.wait_until_ready()
lora_sampling_client_2.wait_until_ready()
full_sampling_client.wait_until_ready()
print("   ✓ Adapter 1 sampling client ready")
print("   ✓ Adapter 2 sampling client ready")
print("   ✓ Full model sampling client ready")
print("=" * 70)

# ============================================================================
# INFERENCE EXAMPLE: Text Generation with Multi-LoRA and Full Model
# ============================================================================
print("\n" + "=" * 70)
print("INFERENCE EXAMPLE: Generating text with trained models")
print("Get Ray Actors:")
ray_actors = service_client.get_ray_actors()
print("Ray actors: ", ray_actors)

print("=" * 70 + "\n")

# Prepare inference prompts for different tasks
geography_prompts = [
    [
        {"role": "system", "content": "You are a geography expert."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
]

math_prompts = [
    [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
]

general_prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about machine learning."},
    ],
]

# Tokenize prompts for inference
geography_formatted = tokenizer.apply_chat_template(geography_prompts, add_generation_prompt=True, tokenize=False)
geography_encoded = tokenizer(geography_formatted, padding=True, return_tensors="pt")

math_formatted = tokenizer.apply_chat_template(math_prompts, add_generation_prompt=True, tokenize=False)
math_encoded = tokenizer(math_formatted, padding=True, return_tensors="pt")

general_formatted = tokenizer.apply_chat_template(general_prompts, add_generation_prompt=True, tokenize=False)
general_encoded = tokenizer(general_formatted, padding=True, return_tensors="pt")

# ============================================================================
# INFERENCE: Test all models with all prompts
# ============================================================================
print("\n" + "=" * 70)
print("INFERENCE: Testing all models with all prompts")
print("=" * 70)

# Combine all prompts for batch inference
all_prompts = geography_prompts + math_prompts + general_prompts
all_formatted = tokenizer.apply_chat_template(all_prompts, add_generation_prompt=True, tokenize=False)
all_encoded = tokenizer(all_formatted, padding=True, return_tensors="pt")

prompt_names = ["Geography", "Math", "General"]
print(f"\nTesting {len(all_prompts)} prompts on 3 models (LoRA 1, LoRA 2, Full Model)")
print("Submitting all requests in parallel...\n")

sampling_params = {"max_new_tokens": 50, "temperature": 0.7}
start_time = time.time()

# Submit ALL requests at once (non-blocking)
lora1_futures = [
    lora_sampling_client.sample(input_ids=all_encoded["input_ids"][i], sampling_params=sampling_params)
    for i in range(len(all_prompts))
]
lora2_futures = [
    lora_sampling_client_2.sample(input_ids=all_encoded["input_ids"][i], sampling_params=sampling_params)
    for i in range(len(all_prompts))
]
full_futures = [
    full_sampling_client.sample(input_ids=all_encoded["input_ids"][i], sampling_params=sampling_params)
    for i in range(len(all_prompts))
]

print(f"✓ Submitted {len(lora1_futures) + len(lora2_futures) + len(full_futures)} requests")
print("Waiting for results...\n")

# Get all results
lora1_results = [f.result() for f in lora1_futures]
lora2_results = [f.result() for f in lora2_futures]
full_results = [f.result() for f in full_futures]

elapsed = time.time() - start_time
print(f"✓ All requests completed in {elapsed:.2f}s\n")

# Print results in a nice format
print("=" * 70)
print("RESULTS")
print("=" * 70)

for i, (prompt, name) in enumerate(zip(all_prompts, prompt_names)):
    print(f"\n{'─' * 70}")
    print(f"Prompt ({name}): {prompt[1]['content']}")
    print(f"{'─' * 70}")
    print(f"  LoRA 1 (task1_attention): {lora1_results[i].output[:100]}...")
    print(f"  LoRA 2 (task2_mlp):       {lora2_results[i].output[:100]}...")
    print(f"  Full Model:               {full_results[i].output[:100]}...")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total prompts: {len(all_prompts)}")
print(f"Total models: 3 (LoRA 1, LoRA 2, Full Model)")
print(f"Total requests: {len(all_prompts) * 3}")
print(f"Total time: {elapsed:.2f}s")
print(f"Avg time per request: {elapsed / (len(all_prompts) * 3):.3f}s")
print("=" * 70)

print("Logprobs:")
print(lora1_results[0].logprobs)
# print(lora2_results[0].logprobs)
# print(full_results[0].logprobs)