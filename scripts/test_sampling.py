from tinker.types import Datum, ModelInput, LoraConfig, TensorData
from tinkerbell.types import ModalDeployConfig
from tinkerbell.client import ServiceClient
import torch

BASE_MODEL = "Qwen/Qwen3-0.6B"
GPU_TYPE = "A100"
NUM_GPUS = 2
CHECKPOINT_PATH_INITIAL = "/tmp/lora-checkpoint-initial"
CHECKPOINT_PATH_CORRUPTED = "/tmp/lora-checkpoint-corrupted"

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE, 
    num_gpus=NUM_GPUS,
    timeout=86400,
    scaledown_window=600,
    max_inputs=200,
    max_wait_time=1200.0,
)

service_client = ServiceClient.deploy_or_connect(deploy_config)

# =============================================================================
# Step 1: Create training client with fresh LoRA weights
# =============================================================================
print("=" * 80)
print("Creating training client with fresh LoRA...")
print("=" * 80)

lora_config = LoraConfig(
    rank=256,
    seed=42,
    train_attn=True,
    train_mlp=True,
    train_unembed=False,
)

training_client = service_client.create_training_client(
    base_model=BASE_MODEL,
    tp_size=1,
    model_name="qwen3-06b-lora-test",
    adapter_name="test_lora",
    lora_config=lora_config.model_dump(),
    model_kwargs={"torch_dtype": "bfloat16"},
)
training_client.wait_until_ready()
print("Training client ready!")

# =============================================================================
# Step 2: Save INITIAL weights and create sampling client
# =============================================================================
print("\n" + "=" * 80)
print("Saving INITIAL LoRA weights and creating sampling client...")
print("=" * 80)

initial_sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path=CHECKPOINT_PATH_INITIAL,
    tp_size=1,
    engine_kwargs={"enable_deterministic_inference": True}
)
initial_sampling_client.wait_until_ready()
print("Initial LoRA sampling client ready!")

# Debug: check LoRA info
print(f"\n--- LoRA Debug Info (INITIAL) ---")
lora_info_initial = initial_sampling_client.get_lora_info()
print(f"SGLang lora_info: {lora_info_initial}")
print(f"client.lora_path: {initial_sampling_client.lora_path}")
print(f"client.adapter_name: {initial_sampling_client.adapter_name}")
print(f"---------------------------------\n")

# =============================================================================
# Step 3: Sample with INITIAL weights
# =============================================================================
tokenizer = training_client.get_tokenizer()

test_conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant. Put your thoughts in <think></think> tags. Respond with <answer></answer> tags."},
        {"role": "user", "content": "Tell me about machine learning."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Put your thoughts in <think></think> tags. Respond with <answer></answer> tags."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
]

formatted_prompts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    for messages in test_conversations
]
tokenized_conversations = [tokenizer.encode(prompt) for prompt in formatted_prompts]

sampling_params = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "sampling_seed": 0,
}

print("\n" + "=" * 80)
print("SAMPLING WITH INITIAL LORA WEIGHTS")
print("=" * 80)

initial_sample_futures = initial_sampling_client.sample_batch(
    batch_kwargs=[
        {
            "sampling_params": sampling_params,
            "input_ids": tokenized_conversation,
        }
    for tokenized_conversation in tokenized_conversations],
)
initial_results = [future.result() for future in initial_sample_futures]

for i, result in enumerate(initial_results):
    print(f"\nPrompt {i+1}: {test_conversations[i][-1]['content']}")
    print(f"Output (first 300 chars): {result.output[:300]}...")

# =============================================================================
# Step 4: Corrupt the LoRA weights
# =============================================================================
print("\n" + "=" * 80)
print("CORRUPTING LoRA weights with garbage training...")
print("=" * 80)

# Create garbage training data
seq_len = 256
batch_size = 4
garbage_datums = []
for _ in range(batch_size):
    input_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,)).tolist()
    labels = torch.randint(0, tokenizer.vocab_size, (seq_len,)).tolist()
    
    datum = Datum(
        model_input=ModelInput.from_ints(input_ids),
        loss_fn_inputs={"labels": TensorData(data=labels, dtype="int64", shape=[len(labels)])}
    )
    garbage_datums.append(datum)

# Do 10 steps with high learning rate to corrupt the weights
for step in range(10):
    training_client.zero_grad().result()
    response = training_client.forward_backward(data=garbage_datums, forward_kwargs={})
    result = response.result()
    training_client.optim_step(
        optimizer_params={"name": "sgd", "lr": 10.0}
    ).result()
    print(f"  Corruption step {step + 1}/10, loss: {result.loss}")

print("LoRA weights corrupted!")

# =============================================================================
# Step 5: Save CORRUPTED weights and reload
# =============================================================================
print("\n" + "=" * 80)
print("Saving CORRUPTED LoRA weights...")
print("=" * 80)

corrupted_sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path=CHECKPOINT_PATH_CORRUPTED,
    tp_size=1,
    engine_kwargs={"enable_deterministic_inference": True}
)
corrupted_sampling_client.wait_until_ready()
print("Corrupted LoRA sampling client ready!")

# Debug: check LoRA info
print(f"\n--- LoRA Debug Info (CORRUPTED) ---")
lora_info_corrupted = corrupted_sampling_client.get_lora_info()
print(f"SGLang lora_info: {lora_info_corrupted}")
print(f"client.lora_path: {corrupted_sampling_client.lora_path}")
print(f"client.adapter_name: {corrupted_sampling_client.adapter_name}")
print(f"-----------------------------------\n")

# =============================================================================
# Step 6: Sample with CORRUPTED weights
# =============================================================================
print("\n" + "=" * 80)
print("SAMPLING WITH CORRUPTED LORA WEIGHTS")
print("=" * 80)

corrupted_sample_futures = corrupted_sampling_client.sample_batch(
    batch_kwargs=[
        {
            "sampling_params": sampling_params,
            "input_ids": tokenized_conversation,
        }
    for tokenized_conversation in tokenized_conversations],
)
corrupted_results = [future.result() for future in corrupted_sample_futures]

for i, result in enumerate(corrupted_results):
    print(f"\nPrompt {i+1}: {test_conversations[i][-1]['content']}")
    print(f"Output (first 300 chars): {result.output[:300]}...")

# =============================================================================
# Step 7: COMPARE INITIAL vs CORRUPTED
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: INITIAL vs CORRUPTED LORA")
print("=" * 80)

for i in range(len(test_conversations)):
    print("\n" + "-" * 60)
    print(f"Test Case {i + 1}: {test_conversations[i][-1]['content']}")
    print("-" * 60)
    
    initial_out = initial_results[i].output
    corrupted_out = corrupted_results[i].output
    
    print(f"\n--- INITIAL LoRA OUTPUT ---")
    print(initial_out[:400] + ("..." if len(initial_out) > 400 else ""))
    
    print(f"\n--- CORRUPTED LoRA OUTPUT ---")
    print(corrupted_out[:400] + ("..." if len(corrupted_out) > 400 else ""))
    
    print(f"\n--- ANALYSIS ---")
    print(f"Initial length: {len(initial_out)} chars")
    print(f"Corrupted length: {len(corrupted_out)} chars")
    
    if initial_out != corrupted_out:
        print("✓ SUCCESS: Outputs are DIFFERENT - LoRA update propagated!")
    else:
        print("✗ FAILURE: Outputs are IDENTICAL - LoRA update NOT working!")
        print("  The sampling server is not using the updated weights!")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print(f"Initial LoRA checkpoint: {CHECKPOINT_PATH_INITIAL}")
print(f"Corrupted LoRA checkpoint: {CHECKPOINT_PATH_CORRUPTED}")
print("\nIf outputs are DIFFERENT, LoRA updating is working correctly!")
print("If outputs are IDENTICAL, there's a bug in LoRA loading/updating.")
print("=" * 80)
