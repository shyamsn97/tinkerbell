import re
import random
import numpy as np
from typing import cast
from datasets import DatasetDict, load_dataset
from tinker.types import Datum, ModelInput, TensorData, LoraConfig
from tinkerbell.types import ModalDeployConfig
from tinkerbell.client import ServiceClient
import wandb

# =============================================================================
# GSM8K GRPO (Group Relative Policy Optimization) Example
# =============================================================================
BASE_MODEL = "Qwen/Qwen3-1.7B"
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
print("GSM8K GRPO Training Example")
print("=" * 60)

# Initialize wandb
wandb.init(
    project="tinkerbell-grpo",
    name=f"gsm8k-grpo-{BASE_MODEL.split('/')[-1]}",
    config={
        "base_model": BASE_MODEL,
        "gpu_type": GPU_TYPE,
        "num_gpus": NUM_GPUS,
        "adapter_name": ADAPTER_NAME,
    },
)

lora_config = LoraConfig(
    rank=128,
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
    initialize_base_model=False,
    model_kwargs={
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
    },
    wait_until_ready=False,
)
training_client.wait_until_ready()

# Create initial sampling client with current (initial) weights
sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path=CHECKPOINT_PATH,
    tp_size=1,
    engine_kwargs={
        "enable_deterministic_inference": True,
        "disable_radix_cache": True,  # CRITICAL: prevent caching when LoRA changes
    }
)
sampling_client.wait_until_ready()
# GRPO hyperparameters
NUM_SAMPLES_PER_PROMPT = 10  # G in GRPO paper - group size
sampling_seeds = [42,43,44,45,46,47,48,49,50] # for reproducibility
NUM_GRPO_STEPS = 100
optimizer_params = {"name": "sgd", "lr": 5e-3}  # Higher LR to see visible changes

# DRASTIC TEST: Use random tokens + cross entropy loss
# If this causes gibberish outputs, training mechanism is confirmed working
USE_RANDOM_TOKENS = False  # Back to real GRPO now that we confirmed training works
RANDOM_SEQ_LEN = 100  # Length of random completion
# Use a conservative range to avoid OOB - stay well under any model's vocab size
RANDOM_TOKEN_MIN = 100
RANDOM_TOKEN_MAX = 30000  # Safe for any modern LLM (most have 32K+ vocab)

sampling_params = {
    "max_new_tokens": 1024,  # Reduce to avoid OOM with large batch
    "temperature": 1.4,  # Lower temp for more coherent outputs
    "top_p": 0.9,
}

# Update wandb config with hyperparameters
wandb.config.update({
    "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
    "num_grpo_steps": NUM_GRPO_STEPS,
    "optimizer": optimizer_params,
    "sampling_params": sampling_params,
    "lora_rank": lora_config.rank,
})


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def extract_token_string(text: str, token: str = "think") -> str:
    """Extract everything between <token> and </token> tags.
    
    Returns the content between the first occurrence of <token> and </token>.
    If no match is found, returns empty string.
    """
    pattern = rf'<{token}>(.*?)</{token}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def compute_gsm8k_reward(output_text: str, ground_truth: str) -> float:
    """Compute reward for GSM8K task based on format and correctness."""
    reward = 0.0
    
    # Reward for proper format (thinking and answer tags)
    if "<think>" in output_text:
        reward += 0.1
    if "</think>" in output_text:
        reward += 0.1
    if "<answer>" in output_text:
        reward += 0.1
    if "</answer>" in output_text:
        reward += 0.1

    # Extract the model's answer from <answer> tags
    model_answer = extract_token_string(output_text, "answer").strip()
    
    # Try to extract numeric answer (remove non-numeric chars except minus and decimal)
    def normalize_answer(ans: str) -> str:
        # Remove commas, dollar signs, percent signs, etc.
        ans = ans.replace(",", "").replace("$", "").replace("%", "").strip()
        # Try to extract just the number
        match = re.search(r'-?\d+\.?\d*', ans)
        if match:
            return match.group()
        return ans.lower()
    
    normalized_model = normalize_answer(model_answer)
    normalized_truth = normalize_answer(ground_truth)

    # Reward for correct answer
    if normalized_model == normalized_truth:
        reward += 2.0
    else:
        # Partial credit: reward for being close (within 10% or small absolute difference)
        try:
            model_num = float(normalized_model)
            truth_num = float(normalized_truth)
            relative_error = abs(model_num - truth_num) / (abs(truth_num) + 1e-8)
            if relative_error < 0.1:
                reward += 0.4  # Within 10%
            elif relative_error < 0.25:
                reward += 0.2  # Within 25%
        except (ValueError, TypeError):
            pass
    
    return reward


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """Compute group-relative advantages for GRPO."""
    rewards_arr = np.array(rewards)
    mean_reward = np.mean(rewards_arr)
    std_reward = np.std(rewards_arr) + 1e-8  # Avoid division by zero
    advantages = (rewards_arr - mean_reward) / std_reward
    return advantages.tolist()


# Load GSM8K dataset
print("Loading GSM8K dataset...")
ds = cast(DatasetDict, load_dataset("openai/gsm8k", name="main"))
train_dataset = ds["train"]
print(f"Loaded {len(train_dataset)} training examples")

# Sample a few examples for training (keep small to fit in memory)
NUM_GSM8K_EXAMPLES = 1
gsm8k_examples = [train_dataset[i] for i in range(NUM_GSM8K_EXAMPLES)]

# Prepare prompts for GSM8K
gsm8k_prompts = []
gsm8k_ground_truths = []

for example in gsm8k_examples:
    question = example["question"]
    answer = example["answer"]
    ground_truth = extract_gsm8k_final_answer(answer)

    prompt = [
        {"role": "system", "content": "You are a helpful math assistant. Think step by step in <think></think> tags. Provide your final answer in <answer></answer> tags."},
        {"role": "user", "content": question},
    ]
    gsm8k_prompts.append(prompt)
    gsm8k_ground_truths.append(ground_truth)

print(f"Prepared {len(gsm8k_prompts)} GSM8K prompts")

# Build inference samples (include_labels=False for sampling)
gsm8k_datums = training_client.build_chat_samples(
    messages=gsm8k_prompts,
    include_labels=False,
)
gsm8k_tokenized = [datum.model_input.to_ints() for datum in gsm8k_datums]

# Track output lengths across steps for debugging
prev_output_lengths = {}  # key: (prompt_idx, sample_idx), value: length
prev_first_output = None  # Track first sample output to detect if weights are updating

# GRPO Training Loop
for step in range(NUM_GRPO_STEPS):
    print(f"\n--- GSM8K GRPO Step {step + 1}/{NUM_GRPO_STEPS} ---")
    
    all_data = []
    all_advantages = []
    curr_output_lengths = {}
    
    for prompt_idx, input_ids in enumerate(gsm8k_tokenized):
        ground_truth = gsm8k_ground_truths[prompt_idx]
        
        if USE_RANDOM_TOKENS:
            # DRASTIC TEST: Generate random training data instead of actual samples
            # We still sample to see if outputs change, but train on random tokens
            print(f"  DEBUG: client lora_path={sampling_client.lora_path}")
            
            # Sample once just to see current model output (for comparison)
            test_sample = sampling_client.sample(
                sampling_params={**sampling_params, "sampling_seed": 42},
                input_ids=input_ids,
            ).result()
            print(f"\n  Prompt {prompt_idx + 1} - Current model output (first 200 chars):")
            print(f"    {test_sample.output[:200]}...")
            
            # Track if output is changing
            first_output_snippet = test_sample.output[:100]
            if prev_first_output is not None:
                if first_output_snippet == prev_first_output:
                    print(f"  ⚠️  WARNING: Output UNCHANGED from previous step!")
                else:
                    print(f"  ✓ Output changed from previous step")
            prev_first_output = first_output_snippet
            
            # Create training data with RANDOM tokens (should cause gibberish if training works)
            for i in range(NUM_SAMPLES_PER_PROMPT):
                # Generate random token IDs for the "completion"
                random_completion = [random.randint(RANDOM_TOKEN_MIN, RANDOM_TOKEN_MAX) for _ in range(RANDOM_SEQ_LEN)]
                
                # Full sequence = prompt + random completion
                full_input_ids = input_ids + random_completion
                
                # Labels: -100 for prompt, random tokens for completion (cross entropy target)
                prompt_len = len(input_ids)
                labels = [-100] * prompt_len + random_completion
                
                datum = Datum(
                    model_input=ModelInput.from_ints(full_input_ids),
                    loss_fn_inputs={
                        "labels": TensorData(data=labels, dtype="int64", shape=[len(labels)]),
                    },
                )
                all_data.append(datum)
                all_advantages.append(0.0)  # Not used for cross entropy
        else:
            # Original GRPO sampling code (commented out for this test)
            sample_futures = []
            # Debug: print the lora_path being used EVERY step
            if prompt_idx == 0:
                print(f"  DEBUG: client lora_path={sampling_client.lora_path}")
                lora_info = sampling_client.get_lora_info()
            for seed in sampling_seeds:
                sample_futures.append(
                    sampling_client.sample(
                        sampling_params={**sampling_params, "sampling_seed": seed},
                        input_ids=input_ids,
                    )
                )
            samples = [f.result() for f in sample_futures]

            # Compute rewards for each sample
            rewards = [
                compute_gsm8k_reward(sample.output, ground_truth) 
                for sample in samples
            ]

            # Print each sample with thinking, answer, and reward
            print(f"\n  Prompt {prompt_idx + 1} (ground truth: {ground_truth}):")
            for i, (sample, reward) in enumerate(zip(samples, rewards)):
                thinking = extract_token_string(sample.output, "think")
                answer = extract_token_string(sample.output, "answer")
                print(f"Sample {i + 1} (reward={reward:.2f}):")

            # Track output lengths and compute diffs
            length_diffs = []
            for i, sample in enumerate(samples):
                curr_len = len(sample.output)
                curr_output_lengths[(prompt_idx, i)] = curr_len
                prev_len = prev_output_lengths.get((prompt_idx, i))
                if prev_len is not None:
                    length_diffs.append(curr_len - prev_len)
                else:
                    length_diffs.append(None)
            
            if step > 0:
                diff_str = [f"{d:+d}" if d is not None else "N/A" for d in length_diffs]
                print(f"  Output length changes from step {step}: {diff_str}")
            print(f"  Current output lengths: {[len(s.output) for s in samples]}")
            # Check if outputs are actually changing (use hash of full output, not just prefix)
            if prompt_idx == 0:
                current_output_hash = hash(samples[0].output)
                if prev_first_output is not None:
                    # Check if ANY length changed
                    any_length_changed = any(d != 0 for d in length_diffs if d is not None)
                    if current_output_hash == prev_first_output and not any_length_changed:
                        print(f"  ⚠️  WARNING: Outputs appear UNCHANGED from previous step!")
                    else:
                        print(f"  ✓ Outputs changing (LoRA updates working)")
                prev_first_output = current_output_hash

            # Compute group-relative advantages
            advantages = compute_grpo_advantages(rewards)
            print(f"  Rewards: {rewards}")
            print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
            print(f"  Advantage stats: min={min(advantages):.4f}, max={max(advantages):.4f}, sum={sum(advantages):.4f}")

            # Create training data for each sample
            for sample, advantage in zip(samples, advantages):
                # Full sequence = prompt + completion
                full_input_ids = input_ids + sample.output_token_ids
                prompt_len = len(input_ids)
                labels = [-100] * prompt_len + sample.output_token_ids
                logprobs_tensor = sample.logprobs.logprobs.to_torch()
                completion_logprobs = logprobs_tensor.tolist()[:len(sample.output_token_ids)]
                sampling_logprobs = [0.0] * prompt_len + completion_logprobs
                advantages_tensor = [0.0] * prompt_len + [advantage] * len(sample.output_token_ids)

                datum = Datum(
                    model_input=ModelInput.from_ints(full_input_ids),
                    loss_fn_inputs={
                        "labels": TensorData(data=labels, dtype="int64", shape=[len(labels)]),
                        "sampling_logprobs": TensorData(data=sampling_logprobs, dtype="float32", shape=[len(sampling_logprobs)]),
                        "advantages": TensorData(data=advantages_tensor, dtype="float32", shape=[len(advantages_tensor)]),
                    },
                )
                all_data.append(datum)
                all_advantages.append(advantage)

    # Perform forward_backward
    # Use cross_entropy for random token test, importance_sampling for normal GRPO
    loss_fn = "cross_entropy" if USE_RANDOM_TOKENS else "importance_sampling"
    print(f"\n  Starting forward_backward with {len(all_data)} samples, loss_fn={loss_fn}...")
    _ = training_client.zero_grad(
        immediate=True,
    ).result()
    result = training_client.forward_backward(
        data=all_data,
        loss_fn=loss_fn,
        return_logprobs=False,
        # zero_grad=True,
        # optimizer_params=optimizer_params,
        immediate=True,
    ).result()
    _ = training_client.optim_step(
        optimizer_params=optimizer_params,
        immediate=True,
    ).result()
    print(f"  Forward_backward complete!")

    print(f"  Loss: {result.loss}")
    sum_gradients = result.sum_gradient

    if sum_gradients is not None:
        # Only print first 5 keys to avoid cluttering output
        limited_gradients = dict(list(sum_gradients.items())[:5])
        print(f"  Sum gradients (first 5): {limited_gradients}")
    else:
        print("Gradient not available")


    # Compute mean reward and advantage stats for this step
    if USE_RANDOM_TOKENS:
        mean_reward = 0.0  # Not applicable for random token test
    else:
        mean_reward = np.mean([compute_gsm8k_reward(s.output, gt) 
                              for s, gt in zip(samples, [gsm8k_ground_truths[0]] * len(samples))])
    mean_advantage = np.mean(all_advantages)
    
    # Prepare wandb log dict
    log_dict = {
        "step": step + 1,
        "loss": result.loss if isinstance(result.loss, float) else np.mean(result.loss),
        "mean_reward": mean_reward,
        "mean_advantage": mean_advantage,
        "use_random_tokens": USE_RANDOM_TOKENS,
    }
    
    # Log sum_gradient dict entries
    if sum_gradients is not None:
        for key, value in sum_gradients.items():
            log_dict[f"grad/{key}"] = value
    
    wandb.log(log_dict)

    # Sync weights to sampling client for next iteration
    # Use step-specific paths to force SGLang to load as new LoRA
    step_checkpoint_path = f"{CHECKPOINT_PATH}-step{step}"
    print(f"  Syncing weights to sampler ({step_checkpoint_path})...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        checkpoint_path=step_checkpoint_path,
        tp_size=1,
        engine_kwargs={
            "enable_deterministic_inference": True,
            "disable_radix_cache": True,
        }
    )
    sampling_client.wait_until_ready()
    # Debug: check what LoRAs are loaded on the server
    lora_info = sampling_client.get_lora_info()
    # print(f"  Weights synced! SGLang lora_info: {lora_info}")
    print(f"  client.lora_path={sampling_client.lora_path}")
    
    # Update previous output lengths for next iteration comparison
    prev_output_lengths = curr_output_lengths.copy()

print("\n" + "=" * 60)
print("GSM8K GRPO Training Complete!")
print("=" * 60)

wandb.finish()

# Clean up - you can save the trained adapter
# training_client.save_checkpoint("/path/to/save/grpo-adapter")
# training_client.push_to_hub("username/grpo-trained-model")
