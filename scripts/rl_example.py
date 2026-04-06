import re
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

sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path=CHECKPOINT_PATH,
    tp_size=1,
    engine_kwargs={
        "enable_deterministic_inference": True,
        "disable_radix_cache": True,
    }
)
sampling_client.wait_until_ready()

# GRPO hyperparameters
NUM_SAMPLES_PER_PROMPT = 9
SAMPLING_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50]
NUM_GRPO_STEPS = 100
OPTIMIZER_PARAMS = {"name": "sgd", "lr": 5e-3}

SAMPLING_PARAMS = {
    "max_new_tokens": 1024,
    "temperature": 1.4,
    "top_p": 0.9,
}

wandb.init(
    project="tinkerbell-grpo",
    name=f"gsm8k-grpo-{BASE_MODEL.split('/')[-1]}",
    config={
        "base_model": BASE_MODEL,
        "gpu_type": GPU_TYPE,
        "num_gpus": NUM_GPUS,
        "adapter_name": ADAPTER_NAME,
        "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
        "num_grpo_steps": NUM_GRPO_STEPS,
        "optimizer": OPTIMIZER_PARAMS,
        "sampling_params": SAMPLING_PARAMS,
        "lora_rank": lora_config.rank,
    },
)


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------

def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final answer following '####' in a GSM8K solution."""
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            return content.replace(",", "").strip()
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>. Returns '' if not found."""
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""


def normalize_number(ans: str) -> str:
    ans = ans.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r'-?\d+\.?\d*', ans)
    return match.group() if match else ans.lower()


def compute_gsm8k_reward(output_text: str, ground_truth: str) -> float:
    reward = 0.0
    if "<think>" in output_text:
        reward += 0.1
    if "</think>" in output_text:
        reward += 0.1
    if "<answer>" in output_text:
        reward += 0.1
    if "</answer>" in output_text:
        reward += 0.1

    model_answer = normalize_number(extract_tag(output_text, "answer"))
    truth = normalize_number(ground_truth)

    if model_answer == truth:
        reward += 2.0
    else:
        try:
            rel_err = abs(float(model_answer) - float(truth)) / (abs(float(truth)) + 1e-8)
            if rel_err < 0.1:
                reward += 0.4
            elif rel_err < 0.25:
                reward += 0.2
        except (ValueError, TypeError):
            pass
    return reward


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    r = np.array(rewards)
    return ((r - r.mean()) / (r.std() + 1e-8)).tolist()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

NUM_GSM8K_EXAMPLES = 1

ds = cast(DatasetDict, load_dataset("openai/gsm8k", name="main"))
train_dataset = ds["train"]

gsm8k_prompts = []
gsm8k_ground_truths = []
for i in range(NUM_GSM8K_EXAMPLES):
    ex = train_dataset[i]
    gt = extract_gsm8k_final_answer(ex["answer"])
    gsm8k_prompts.append([
        {"role": "system", "content": "You are a helpful math assistant. Think step by step in <think></think> tags. Provide your final answer in <answer></answer> tags."},
        {"role": "user", "content": ex["question"]},
    ])
    gsm8k_ground_truths.append(gt)

gsm8k_datums = training_client.build_chat_samples(
    messages=gsm8k_prompts,
    include_labels=False,
)
gsm8k_tokenized = [datum.model_input.to_ints() for datum in gsm8k_datums]

# ---------------------------------------------------------------------------
# GRPO Training Loop
# ---------------------------------------------------------------------------

prev_output_hash = None

for step in range(NUM_GRPO_STEPS):
    print(f"\n--- GRPO Step {step + 1}/{NUM_GRPO_STEPS} ---")

    all_data: list[Datum] = []
    all_advantages: list[float] = []

    for prompt_idx, input_ids in enumerate(gsm8k_tokenized):
        ground_truth = gsm8k_ground_truths[prompt_idx]

        # Sample completions in parallel
        futures = [
            sampling_client.sample(
                sampling_params={**SAMPLING_PARAMS, "sampling_seed": seed},
                input_ids=input_ids,
            )
            for seed in SAMPLING_SEEDS
        ]
        samples = [f.result() for f in futures]

        # Rewards & advantages
        rewards = [compute_gsm8k_reward(s.output, ground_truth) for s in samples]
        advantages = compute_grpo_advantages(rewards)

        # Log per-prompt info
        print(f"  Prompt {prompt_idx + 1} (gt={ground_truth})")
        print(f"    Rewards:    {rewards}")
        print(f"    Advantages: {[f'{a:.3f}' for a in advantages]}")
        print(f"    Lengths:    {[len(s.output) for s in samples]}")

        # Detect output staleness
        if prompt_idx == 0:
            h = hash(samples[0].output)
            if prev_output_hash is not None:
                status = "✓ changed" if h != prev_output_hash else "⚠ UNCHANGED"
                print(f"    Output: {status}")
            prev_output_hash = h

        # Build Datum for each sample
        for sample, adv in zip(samples, advantages):
            prompt_len = len(input_ids)
            full_ids = input_ids + sample.output_token_ids
            labels = [-100] * prompt_len + sample.output_token_ids

            logprobs_list = sample.logprobs.logprobs.to_torch().tolist()[:len(sample.output_token_ids)]
            sampling_lp = [0.0] * prompt_len + logprobs_list
            adv_seq = [0.0] * prompt_len + [adv] * len(sample.output_token_ids)

            all_data.append(Datum(
                model_input=ModelInput.from_ints(full_ids),
                loss_fn_inputs={
                    "labels": TensorData(data=labels, dtype="int64", shape=[len(labels)]),
                    "sampling_logprobs": TensorData(data=sampling_lp, dtype="float32", shape=[len(sampling_lp)]),
                    "advantages": TensorData(data=adv_seq, dtype="float32", shape=[len(adv_seq)]),
                },
            ))
            all_advantages.append(adv)

    # Train step: zero_grad → forward_backward → optim_step
    training_client.zero_grad(immediate=True).result()
    result = training_client.forward_backward(
        data=all_data,
        loss_fn="importance_sampling",
        return_logprobs=False,
        immediate=True,
    ).result()
    training_client.optim_step(
        optimizer_params=OPTIMIZER_PARAMS,
        immediate=True,
    ).result()

    # Logging
    loss_val = result.loss if isinstance(result.loss, float) else np.mean(result.loss)
    mean_reward = np.mean([compute_gsm8k_reward(s.output, gsm8k_ground_truths[0]) for s in samples])
    print(f"  Loss: {loss_val:.4f}  Mean reward: {mean_reward:.2f}")

    log_dict = {
        "step": step + 1,
        "loss": loss_val,
        "mean_reward": mean_reward,
        "mean_advantage": np.mean(all_advantages),
    }
    if result.sum_gradient:
        for k, v in result.sum_gradient.items():
            log_dict[f"grad/{k}"] = v
    wandb.log(log_dict)

    # Sync weights → sampler for next step
    step_ckpt = f"{CHECKPOINT_PATH}-step{step}"
    sampling_client = training_client.save_weights_and_get_sampling_client(
        checkpoint_path=step_ckpt,
        tp_size=1,
        engine_kwargs={
            "enable_deterministic_inference": True,
            "disable_radix_cache": True,
        },
    )
    sampling_client.wait_until_ready()

print("\n" + "=" * 60)
print("GSM8K GRPO Training Complete!")
print("=" * 60)

wandb.finish()
