import numpy as np
from tinkerbell.types import ModalDeployConfig, LoraConfig
from tinkerbell.types.datum import Datum
from tinkerbell.types.model_input import ModelInput
from tinkerbell.client import ServiceClient

# =============================================================================
# GRPO (Group Relative Policy Optimization) Example
# =============================================================================
BASE_MODEL = "Qwen/Qwen3-0.6B"
GPU_TYPE = "A100"
NUM_GPUS = 2
ADAPTER_NAME = "grpo-adapter"
CHECKPOINT_PATH = "/tmp/grpo-checkpoint"

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE, 
    num_gpus=NUM_GPUS,
    timeout=86400,
    container_idle_timeout=600,
    max_inputs=200,
    max_wait_time=1200.0,
)

service_client = ServiceClient.deploy_or_connect(deploy_config)

print("=" * 60)
print("GRPO Training Example")
print("=" * 60)

lora_config = LoraConfig(
    rank=64,
    train_unembed=False,
    train_mlp=True,
    train_attn=True,
)

training_client = service_client.create_training_client(
    base_model=BASE_MODEL,
    tp_size=1,
    model_name="qwen3-06b-grpo",
    adapter_name=ADAPTER_NAME,
    lora_config=lora_config.model_dump(),
    initialize_random_weights=False,
    model_kwargs={
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
    },
    wait_until_ready=False,
)
training_client.wait_until_ready()

# Create initial sampling client with current (initial) weights
grpo_sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path=CHECKPOINT_PATH,
    tp_size=1,
)

# GRPO hyperparameters
NUM_SAMPLES_PER_PROMPT = 4  # G in GRPO paper
GRPO_BETA = 0.1  # KL penalty coefficient

sampling_params = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Define prompts for GRPO training
grpo_prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant. Put your thoughts in <think></think> tags. Respond with <answer></answer> tags."},
        {"role": "user", "content": "What is 2 + 3?"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Put your thoughts in <think></think> tags. Respond with <answer></answer> tags."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
]

# Tokenize prompts
tokenizer = grpo_sampling_client.get_tokenizer()
grpo_formatted = [
    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for msgs in grpo_prompts
]
grpo_tokenized = [tokenizer.encode(prompt) for prompt in grpo_formatted]


def compute_reward(output_text: str, prompt_idx: int) -> float:
    """Simple reward function based on format and correctness."""
    reward = 0.0
    
    # Reward for proper format
    if "<think>" in output_text and "</think>" in output_text:
        reward += 0.3
    if "<answer>" in output_text and "</answer>" in output_text:
        reward += 0.3
    
    # Reward for correct answer (simple heuristics)
    if prompt_idx == 0:  # "What is 2 + 3?"
        if "5" in output_text:
            reward += 0.4
    elif prompt_idx == 1:  # "What is the capital of France?"
        if "paris" in output_text.lower():
            reward += 0.4
    
    return reward


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """Compute group-relative advantages for GRPO."""
    rewards_arr = np.array(rewards)
    mean_reward = np.mean(rewards_arr)
    std_reward = np.std(rewards_arr) + 1e-8  # Avoid division by zero
    advantages = (rewards_arr - mean_reward) / std_reward
    return advantages.tolist()


# GRPO Training Loop
NUM_GRPO_STEPS = 3
optimizer_params = {"name": "adam", "lr": 1e-4}

for step in range(NUM_GRPO_STEPS):
    print(f"\n--- GRPO Step {step + 1}/{NUM_GRPO_STEPS} ---")
    
    all_data = []
    all_advantages = []
    
    for prompt_idx, input_ids in enumerate(grpo_tokenized):
        # Sample multiple completions for this prompt (using updated policy)
        sample_futures = grpo_sampling_client.sample_batch(
            batch_kwargs=[
                {
                    "sampling_params": sampling_params,
                    "input_ids": input_ids,
                }
                for _ in range(NUM_SAMPLES_PER_PROMPT)
            ],
        )
        samples = [f.result() for f in sample_futures]
        
        # Compute rewards for each sample
        rewards = [
            compute_reward(sample.output, prompt_idx) 
            for sample in samples
        ]
        print(f"  Prompt {prompt_idx + 1} rewards: {rewards}")
        
        # Compute group-relative advantages
        advantages = compute_grpo_advantages(rewards)
        print(f"  Prompt {prompt_idx + 1} advantages: {[f'{a:.3f}' for a in advantages]}")
        
        # Create training data for each sample
        for sample, advantage in zip(samples, advantages):
            # Full sequence = prompt + completion
            full_input_ids = input_ids + sample.output_token_ids
            
            # Create attention mask (1 for all tokens)
            attention_mask = [1] * len(full_input_ids)
            
            # Labels: use -100 (MASK_TOKEN_ID) for prompt tokens to ignore them in loss
            # Only compute loss on completion tokens
            prompt_len = len(input_ids)
            labels = [-100] * prompt_len + sample.output_token_ids
            
            # Sampling logprobs: pad prompt positions with 0 (they'll be masked anyway)
            # sample.logprobs.logprobs is TensorData with shape [completion_len]
            logprobs_tensor = sample.logprobs.logprobs.to_torch()
            completion_logprobs = logprobs_tensor.tolist()
            sampling_logprobs = [0.0] * prompt_len + completion_logprobs
            
            # Advantages: same value for all completion tokens, 0 for prompt tokens
            advantages_tensor = [0.0] * prompt_len + [advantage] * len(sample.output_token_ids)
            
            datum = Datum(
                model_input=ModelInput(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                ),
                loss_fn_inputs={
                    "labels": labels,
                    "sampling_logprobs": sampling_logprobs,
                    "advantages": advantages_tensor,
                },
            )
            all_data.append(datum)
            all_advantages.append(advantage)
    
    # Perform forward_backward with importance_sampling loss (GRPO-style)
    result = training_client.forward_backward(
        data=all_data,
        loss_fn="importance_sampling",
        return_logprobs=True,
        zero_grad=True,
        optimizer_params=optimizer_params,
        immediate=True,
    ).result()
    
    print(f"  Loss: {result.loss}")
    # print(f"  Gradient norm: {result.sum_gradient}" if result.sum_gradient else "")
    
    # Sync weights to sampling client for next iteration
    grpo_sampling_client = training_client.save_weights_and_get_sampling_client(
        checkpoint_path=CHECKPOINT_PATH,
        tp_size=1,
    )
    print(f"  Weights synced to sampler")

print("\n" + "=" * 60)
print("GRPO Training Complete!")
print("=" * 60)

# Clean up - you can save the trained adapter
# training_client.save_checkpoint("/path/to/save/grpo-adapter")
# training_client.push_to_hub("username/grpo-trained-model")
