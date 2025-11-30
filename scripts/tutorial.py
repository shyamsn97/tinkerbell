from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig, LoraConfig
from tinkerbell.renderer import Renderer, TrainOnWhat
from tqdm import tqdm
import time
import asyncio
import concurrent.futures


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

training_client = service_client.create_training_client(
    model_id=MODEL_NAME,
    model_name="multi_lora_model",  # Give it a name for multi-adapter support
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=False,
    lora_config=lora_config_1.model_dump(),
    adapter_name="task1_attention",  # Name the first adapter
    model_kwargs={
        "torch_dtype": "bfloat16",
    }
)
training_client.wait_until_ready()
print("   ✓ Adapter 'task1_attention' ready")

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

# IMPORTANT: Use the SAME model_name to share the actor group!
# This is what makes multi-LoRA work - multiple adapters on one base model
training_client_2 = service_client.create_training_client(
    model_id=MODEL_NAME,
    model_name="multi_lora_model",  # SAME model name = shared actor group!
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=False,
    lora_config=lora_config_2.model_dump(),
    adapter_name="task2_mlp",  # Name the second adapter
    model_kwargs={
        "torch_dtype": "bfloat16",
    }
)
training_client_2.wait_until_ready()
print("   ✓ Adapter 'task2_mlp' ready (sharing actor with task1_attention)")
print(f"\n✓ Both LoRA adapters sharing the same base model actor group!")
print(f"  - Client 1: 'task1_attention' (rank={lora_config_1.rank}, attn layers)")
print(f"  - Client 2: 'task2_mlp' (rank={lora_config_2.rank}, mlp layers)")
print(f"  - Total GPU usage: {2} GPUs (tp_size=2) for both adapters combined")
print("=" * 70)

print("\n3. Creating full model training client (no LoRA)...")
full_model_client = service_client.create_training_client(
    model_id=MODEL_NAME,
    model_name="full_model",
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=False,
    lora_config=None,  # No LoRA = train full model
    model_kwargs={
        "torch_dtype": "bfloat16",
    }
)
full_model_client.wait_until_ready()
print("   ✓ Full model client ready")
print("=" * 70)

# ============================================================================
# TRAINING EXAMPLE: Multi-Adapter Training with Separate Clients
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING EXAMPLE: Multi-adapter training with separate clients")
print("Get Ray Actors:")
ray_actors_response = service_client.get_ray_actors()
ray_actors = ray_actors_response.result()
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
task1_data = renderer.build_chat_examples(
    conversations=task1_conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

task2_data = renderer.build_chat_examples(
    conversations=task2_conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

full_model_data = renderer.build_chat_examples(
    conversations=full_model_conversations,
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
print("\n1. Saving adapter 1 (task1_attention)...")
lora_sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/task1-attention-qwen",
    tp_size=1,
    wait_until_ready=True
)
print("   ✓ Adapter 1 sampling client ready")

# Save adapter 2
print("\n2. Saving adapter 2 (task2_mlp)...")
lora_sampling_client_2 = training_client_2.save_weights_and_get_sampling_client(
    checkpoint_path="/models/task2-mlp-qwen",
    tp_size=1,
    wait_until_ready=True
)
print("   ✓ Adapter 2 sampling client ready")

# Save full model
print("\n3. Saving full model (no LoRA)...")
full_sampling_client = full_model_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/full-qwen",
    tp_size=1,
    wait_until_ready=True
)
print("   ✓ Full model sampling client ready")
print("=" * 70)

# ============================================================================
# INFERENCE EXAMPLE: Text Generation with Multi-LoRA and Full Model
# ============================================================================
print("\n" + "=" * 70)
print("INFERENCE EXAMPLE: Generating text with trained models")
print("Get Ray Actors:")
ray_actors_response = service_client.get_ray_actors()
ray_actors = ray_actors_response.result()
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
# Multi-LoRA Inference: Using separate adapter clients
# ============================================================================
print("\n" + "=" * 70)
print("DEBUG: Checking Ray actors before inference")
print("=" * 70)
ray_actors_response = service_client.get_ray_actors().result()
print(f"Ray Actors: {ray_actors_response}")
print("=" * 70)

print("\n" + "=" * 70)
print("MULTI-LORA INFERENCE: Testing with separate adapter clients")
print("=" * 70)

print("\n1. Geography question with Adapter 1 (task1_attention):")
print("   Prompt:", geography_prompts[0][1]["content"])
start_time = time.time()
geography_output = lora_sampling_client.sample(
    input_ids=geography_encoded["input_ids"][0],
    sampling_params={
        "max_new_tokens": 50,
        "temperature": 0.7,
    },
).result()
print(f"   Response: {geography_output.outputs[0]}")
print(f"   Time: {time.time() - start_time:.2f}s")

print("\n2. Math question with Adapter 2 (task2_mlp):")
print("   Prompt:", math_prompts[0][1]["content"])
start_time = time.time()
math_output = lora_sampling_client_2.sample(
    input_ids=math_encoded["input_ids"][0],
    sampling_params={
        "max_new_tokens": 50,
        "temperature": 0.7,
    },
).result()
print(f"   Response: {math_output.outputs[0]}")
print(f"   Time: {time.time() - start_time:.2f}s")

# ============================================================================
# Full Model Inference: No LoRA adapters
# ============================================================================
print("\n" + "=" * 70)
print("FULL MODEL INFERENCE: Testing without LoRA")
print("=" * 70)

print("\n1. General question with full model:")
print("   Prompt:", general_prompts[0][1]["content"])
start_time = time.time()
general_full_output = full_sampling_client.sample(
    input_ids=general_encoded["input_ids"][0],
    sampling_params={
        "max_new_tokens": 50,
        "temperature": 0.7,
    },
).result()
print(f"   Response: {general_full_output.outputs[0]}")
print(f"   Time: {time.time() - start_time:.2f}s")

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print("\nMulti-LoRA Model (Separate Clients):")
print("  - Each adapter has its own training client")
print("  - Memory efficient: only trains low-rank matrices")
print("  - Client 1: task1_attention (rank=8, attn)")
print("  - Client 2: task2_mlp (rank=4, mlp)")
print("\nFull Model:")
print("  - Trains all parameters")
print("  - Higher memory usage but potentially better quality")
print("  - No adapter management needed")
print("=" * 70)


# ============================================================================
# BATCH INFERENCE EXAMPLES: Multithreaded and Async
# ============================================================================
print("\n" + "=" * 70)
print("BATCH INFERENCE: Testing throughput with multiple requests")
print("=" * 70)

# Create a batch of mixed prompts
batch_prompts = geography_prompts + math_prompts + general_prompts
batch_formatted = tokenizer.apply_chat_template(batch_prompts, add_generation_prompt=True, tokenize=False)
batch_encoded = tokenizer(batch_formatted, padding=True, return_tensors="pt")

print(f"\nBatch size: {len(batch_prompts)} prompts")

# Test 1: Multithreaded inference with Adapter 1
print("\n" + "=" * 70)
print("Test 1: Multithreaded inference with Adapter 1 (task1_attention)")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    tinkerbell_futures = [
        lora_sampling_client.sample(
            input_ids=batch_encoded["input_ids"][i],
            sampling_params={"max_new_tokens": 50, "temperature": 0.7}
        ) for i in range(len(batch_prompts))
    ]
    thread_futures = [executor.submit(lambda f: f.result(), tf) for tf in tinkerbell_futures]
    outputs_list = []
    for future in tqdm(concurrent.futures.as_completed(thread_futures), total=len(thread_futures), desc="Adapter 1 requests"):
        outputs = future.result()
        outputs_list.append(outputs)
print(f"✓ Completed {len(outputs_list)} requests in {time.time() - start_time:.2f}s")
print("=" * 70)

# Test 2: Multithreaded inference with Full Model
print("\nTest 2: Multithreaded Full Model inference")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    tinkerbell_futures = [
        full_sampling_client.sample(
            input_ids=batch_encoded["input_ids"][i],
            sampling_params={"max_new_tokens": 50, "temperature": 0.7}
        ) for i in range(len(batch_prompts))
    ]
    thread_futures = [executor.submit(lambda f: f.result(), tf) for tf in tinkerbell_futures]
    outputs_list = []
    for future in tqdm(concurrent.futures.as_completed(thread_futures), total=len(thread_futures), desc="Full model requests"):
        outputs = future.result()
        outputs_list.append(outputs)
print(f"✓ Completed {len(outputs_list)} requests in {time.time() - start_time:.2f}s")
print("=" * 70)

# Test 3: Async inference with Adapter 1 (two-phase approach)
print("\n" + "=" * 70)
print("Test 3: Async inference with Adapter 1 (two-phase)")
print("=" * 70)

async def async_sample_two_phase():
    """
    Two-phase async sampling for maximum efficiency:
    Phase 1: Submit all requests in parallel (first await)
    Phase 2: Poll all results in parallel (second await)
    """
    start_time = time.time()
    
    # Phase 1: Submit all requests and get futures
    print(f"Phase 1: Submitting {len(batch_prompts)} async requests...")
    submit_start = time.time()
    
    async def submit_request(i):
        """First await: submit request and return the future."""
        return await lora_sampling_client.sample_async(
            input_ids=batch_encoded["input_ids"][i],
            sampling_params={"max_new_tokens": 50, "temperature": 0.7}
        )
    
    # Gather all futures (first await for each)
    futures = await asyncio.gather(*[submit_request(i) for i in range(len(batch_prompts))])
    print(f"  ✓ All requests submitted in {time.time() - submit_start:.2f}s")
    
    # Phase 2: Poll all futures for results
    print(f"Phase 2: Polling {len(futures)} futures for results...")
    poll_start = time.time()
    
    # Gather all results (second await for each)
    outputs_list = await asyncio.gather(*futures)
    print(f"  ✓ All results received in {time.time() - poll_start:.2f}s")
    
    elapsed = time.time() - start_time
    print(f"✓ All {len(outputs_list)} async requests completed!")
    print(f"Total time: {elapsed:.2f} seconds")
    print("=" * 70)
    return outputs_list


# Run the async function
asyncio.run(async_sample_two_phase())
# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ TUTORIAL COMPLETE!")
print("=" * 70)
print("\n📚 What we demonstrated:")
print("\n1. Multi-LoRA Training:")
print("   ✓ Created 2 LoRA adapters with separate training clients")
print("   ✓ Adapter 1: High-rank (8), attention-focused")
print("   ✓ Adapter 2: Low-rank (4), MLP-focused")
print("   ✓ Each adapter trained independently with its own client")
print("\n2. Full Model Training:")
print("   ✓ Trained without LoRA for comparison")
print("   ✓ All parameters updated (vs frozen base + LoRA)")
print("\n3. Multi-LoRA Inference:")
print("   ✓ Used separate sampling clients for each adapter")
print("   ✓ Task-specific inference with appropriate adapter clients")
print("\n4. Full Model Inference:")
print("   ✓ Standard inference without adapter selection")
print("\n5. Performance Testing:")
print("   ✓ Multithreaded inference with ThreadPoolExecutor")
print("   ✓ Async inference with two-phase approach")
print("   ✓ Compared throughput between adapters and Full Model")
print("\n🎯 Key Takeaways:")
print("  • Multi-LoRA enables task-specific fine-tuning")
print("  • Separate clients allow independent adapter training")
print("  • Memory efficient: ~1-2% overhead per adapter")
print("  • Full model training for comparison and baseline")
print("  • Flexible inference: use different adapter clients")
print("\n" + "=" * 70)

