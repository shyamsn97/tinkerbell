"""GSM8K GRPO (Group Relative Policy Optimization) example.

Per step:
  1. Sample N completions per prompt with the current sampler.
  2. Score each completion → reward.
  3. Compute group-relative advantages: reward - mean(rewards_in_group).
  4. forward_backward with `importance_sampling` loss; optim_step.
  5. Snapshot weights → spin a fresh sampling client for the next step.
"""

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import cast

# Prefer the checkout's `tinkerbell/` package over any previously installed
# wheel when this example is run as `python scripts/grpo_example.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wandb
from datasets import DatasetDict, load_dataset
from tinker.types import Datum, LoraConfig, ModelInput, TensorData

from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig

BASE_MODEL = "Qwen/Qwen3-1.7B"
GPU_TYPE = "A100"
NUM_GPUS = 2
TRAIN_TP_SIZE = 1
SAMPLING_TP_SIZE = 1
ADAPTER_NAME = "grpo-adapter"
CHECKPOINT_ROOT = "/tmp/grpo-checkpoint"

# Match the Tinker cookbook RL tutorial shape: 16 problems per step, 8
# completions per problem, mean-centered group advantages.
NUM_GSM8K_EXAMPLES = int(os.getenv("NUM_GSM8K_EXAMPLES", "16"))
NUM_SAMPLES_PER_PROMPT = int(os.getenv("NUM_SAMPLES_PER_PROMPT", "8"))
NUM_GRPO_STEPS = int(os.getenv("NUM_GRPO_STEPS", "100"))

# Dynamic oversampling: keep pulling fresh prompt batches each step until we
# have at least TARGET_LIVE_GROUPS groups with non-degenerate advantages, or
# until we've drawn MAX_BATCHES_PER_STEP batches. Set TARGET_LIVE_GROUPS=0 to
# disable and use a single fixed batch (old behavior).
TARGET_LIVE_GROUPS = int(os.getenv("TARGET_LIVE_GROUPS", "8"))
MAX_BATCHES_PER_STEP = int(os.getenv("MAX_BATCHES_PER_STEP", "4"))

# Microbatch size for the training side: a single forward_backward processes
# at most this many rollouts. Smaller = less peak activation memory; multiple
# microbatches accumulate gradients for the same effective batch size. Tune
# this if you OOM (drop) or have memory headroom (raise). 8 is a safe start
# on a 40GB A100 with 1.7B + LoRA rank 128 + grad checkpointing on.
MICROBATCH_SIZE = int(os.getenv("MICROBATCH_SIZE", "4"))

OPTIMIZER_PARAMS = {"name": "adam", "lr": 4e-5, "betas": [0.9, 0.95]}
LOSS_FN = "importance_sampling"
LOSS_FN_CONFIG = {}

SAMPLING_PARAMS = {
    "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "2048")),
    "temperature": 1.0,
    "top_p": 0.95,
}

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE,
    num_gpus=NUM_GPUS,
    timeout=86400,
    scaledown_window=600,
    max_inputs=200,
    max_wait_time=1200.0,
)

service_client = ServiceClient.deploy_or_connect(deploy_config, redeploy=True)

lora_config = LoraConfig(
    rank=128,
    train_unembed=False,
    train_mlp=True,
    train_attn=True,
)

training_client = service_client.create_training_client(
    base_model=BASE_MODEL,
    tp_size=TRAIN_TP_SIZE,
    model_name="qwen3-06b-grpo",
    adapter_name=ADAPTER_NAME,
    lora_config=lora_config.model_dump(),
    initialize_base_model=False,
    model_kwargs={
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,
    },
)
training_client.wait_until_ready()

sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path=f"{CHECKPOINT_ROOT}-init",
    tp_size=SAMPLING_TP_SIZE,
    engine_kwargs={
        "disable_radix_cache": True,
    },
)
sampling_client.wait_until_ready()

wandb.init(
    project="tinkerbell-grpo",
    name=f"gsm8k-grpo-{BASE_MODEL.split('/')[-1]}",
    settings=wandb.Settings(x_save_requirements=False, disable_code=True),
    config={
        "base_model": BASE_MODEL,
        "gpu_type": GPU_TYPE,
        "num_gpus": NUM_GPUS,
        "train_tp_size": TRAIN_TP_SIZE,
        "sampling_tp_size": SAMPLING_TP_SIZE,
        "adapter_name": ADAPTER_NAME,
        "num_gsm8k_examples": NUM_GSM8K_EXAMPLES,
        "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
        "num_grpo_steps": NUM_GRPO_STEPS,
        "optimizer": OPTIMIZER_PARAMS,
        "loss_fn": LOSS_FN,
        "loss_fn_config": LOSS_FN_CONFIG,
        "sampling_params": SAMPLING_PARAMS,
        "lora_rank": lora_config.rank,
    },
)


# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------


def extract_gsm8k_final_answer(text: str) -> str:
    """Pull the final answer that follows '####' in a GSM8K solution."""
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].lstrip(":").strip()
            return content.replace(",", "").strip()
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def normalize_number(ans: str) -> str:
    ans = ans.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+\.?\d*", ans)
    return match.group() if match else ans.lower()


def compute_gsm8k_reward(output_text: str, ground_truth: str) -> float:
    """Binary verifier reward, matching the Tinker cookbook example."""
    answer = extract_boxed(output_text)
    if answer is None:
        return 0.0
    return 1.0 if normalize_number(answer) == normalize_number(ground_truth) else 0.0


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    mean_reward = sum(rewards) / len(rewards)
    return [reward - mean_reward for reward in rewards]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_grpo_datum(
    input_ids: list[int],
    output_token_ids: list[int],
    output_logprobs: list[float],
    advantage: float,
) -> Datum:
    """Construct a Datum for one rollout, with HF-style shifted labels.

    `logprobs[j]` predicts `full_ids[j+1]`, so the prompt-mask region is
    `prompt_len - 1` positions; the last position predicts EOS and is masked.
    """
    prompt_len = len(input_ids)
    full_ids = input_ids + output_token_ids
    seq_len = len(full_ids)

    labels = [-100] * (prompt_len - 1) + list(output_token_ids) + [-100]
    sampling_lp = [0.0] * (prompt_len - 1) + list(output_logprobs) + [0.0]
    adv_seq = [0.0] * (prompt_len - 1) + [advantage] * len(output_token_ids) + [0.0]
    assert len(labels) == seq_len == len(sampling_lp) == len(adv_seq)

    return Datum(
        model_input=ModelInput.from_ints(full_ids),
        loss_fn_inputs={
            "labels": TensorData(data=labels, dtype="int64", shape=[seq_len]),
            "sampling_logprobs": TensorData(
                data=sampling_lp, dtype="float32", shape=[seq_len]
            ),
            "advantages": TensorData(data=adv_seq, dtype="float32", shape=[seq_len]),
        },
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

ds = cast(DatasetDict, load_dataset("openai/gsm8k", name="main"))
train_dataset = ds["train"]

QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
FEWSHOT_PREFIX = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
]


def batch_indices(start: int, n: int) -> list[int]:
    return [(start + i) % len(train_dataset) for i in range(n)]


def sample_gsm8k_batch(start: int) -> tuple[list[int], list[list[int]], list[str]]:
    indices = batch_indices(start, NUM_GSM8K_EXAMPLES)
    prompts: list[list[dict[str, str]]] = []
    ground_truths: list[str] = []
    for idx in indices:
        ex = train_dataset[idx]
        prompts.append(
            [
                *FEWSHOT_PREFIX,
                {"role": "user", "content": ex["question"] + QUESTION_SUFFIX},
            ]
        )
        ground_truths.append(extract_gsm8k_final_answer(ex["answer"]))

    datums = training_client.build_chat_samples(messages=prompts, include_labels=False)
    return indices, [d.model_input.to_ints() for d in datums], ground_truths


# ---------------------------------------------------------------------------
# GRPO loop
# ---------------------------------------------------------------------------

prev_step_ckpt = f"{CHECKPOINT_ROOT}-init"
dataset_cursor = 0

for step in range(NUM_GRPO_STEPS):
    print(f"\n--- GRPO Step {step + 1}/{NUM_GRPO_STEPS} ---")

    all_data: list[Datum] = []
    all_rewards: list[float] = []
    all_advantages: list[float] = []
    prompt_mean_rewards: list[float] = []
    # Track first-batch reward separately: this is unconfounded by dynamic
    # oversampling (we always draw at least one batch off the same cursor),
    # so it's the most reliable progress signal in W&B.
    first_batch_rewards: list[float] = []
    truncated_count = 0
    live_groups = 0
    dead_groups = 0
    total_rollouts = 0
    sample_secs = 0.0
    batches_drawn = 0

    while batches_drawn < max(1, MAX_BATCHES_PER_STEP):
        prompt_indices, gsm8k_tokenized, gsm8k_ground_truths = sample_gsm8k_batch(
            dataset_cursor
        )
        dataset_cursor = (dataset_cursor + NUM_GSM8K_EXAMPLES) % len(train_dataset)
        batches_drawn += 1
        print(f"  Batch {batches_drawn} indices {prompt_indices[:3]}...{prompt_indices[-3:]}")

        sample_start = time.time()
        batch_kwargs = [
            {
                "sampling_params": SAMPLING_PARAMS,
                "input_ids": input_ids,
            }
            for input_ids in gsm8k_tokenized
            for _ in range(NUM_SAMPLES_PER_PROMPT)
        ]
        flat_samples = sampling_client.sample_many(batch_kwargs).result()
        samples_per_prompt = [
            flat_samples[i : i + NUM_SAMPLES_PER_PROMPT]
            for i in range(0, len(flat_samples), NUM_SAMPLES_PER_PROMPT)
        ]
        if len(samples_per_prompt) != len(gsm8k_tokenized):
            raise RuntimeError(
                f"sample_many returned {len(flat_samples)} samples for "
                f"{len(gsm8k_tokenized)} prompts x {NUM_SAMPLES_PER_PROMPT}"
            )
        batch_secs = time.time() - sample_start
        sample_secs += batch_secs
        batch_rollouts = NUM_GSM8K_EXAMPLES * NUM_SAMPLES_PER_PROMPT
        total_rollouts += batch_rollouts
        print(
            f"    Sampled {batch_rollouts} rollouts in {batch_secs:.1f}s "
            f"({batch_rollouts / batch_secs:.1f} rollouts/s)"
        )

        for prompt_idx, (input_ids, samples) in enumerate(
            zip(gsm8k_tokenized, samples_per_prompt)
        ):
            ground_truth = gsm8k_ground_truths[prompt_idx]
            rewards = [compute_gsm8k_reward(s.output, ground_truth) for s in samples]
            advantages = compute_grpo_advantages(rewards)
            prompt_mean_rewards.append(sum(rewards) / len(rewards))
            all_rewards.extend(rewards)
            if batches_drawn == 1:
                first_batch_rewards.extend(rewards)
            for s in samples:
                if s.finish_reason and "length" in str(s.finish_reason).lower():
                    truncated_count += 1

            if all(adv == 0.0 for adv in advantages):
                dead_groups += 1
                continue

            live_groups += 1
            print(f"    Prompt (gt={ground_truth})")
            print(f"      Rewards:    {[f'{r:.1f}' for r in rewards]}")
            print(f"      Advantages: {[f'{a:.3f}' for a in advantages]}")
            print(f"      Lengths:    {[len(s.output) for s in samples]}")

            for sample, adv in zip(samples, advantages):
                if sample.logprobs is None or sample.logprobs.logprobs is None:
                    meta_keys = sorted((sample.meta_info or {}).keys())
                    raise RuntimeError(
                        "Sampler did not return output logprobs; GRPO requires "
                        f"return_logprob=True/top_logprobs_num=1. meta_info keys={meta_keys}"
                    )
                sampler_lp = sample.logprobs.logprobs.to_torch().tolist()
                # Sampler/tokenizer can disagree on the final token (e.g. EOS
                # accounting); clip both sides to the common prefix length.
                n = min(len(sample.output_token_ids), len(sampler_lp))
                if n == 0:
                    continue
                all_data.append(
                    build_grpo_datum(
                        input_ids=input_ids,
                        output_token_ids=sample.output_token_ids[:n],
                        output_logprobs=sampler_lp[:n],
                        advantage=adv,
                    )
                )
                all_advantages.append(adv)

        if live_groups >= TARGET_LIVE_GROUPS:
            break

    print(
        f"  Signal density: {live_groups} live groups, {dead_groups} degenerate, "
        f"{len(all_data)} rollouts to train on (out of {total_rollouts}, "
        f"{batches_drawn} batch{'es' if batches_drawn > 1 else ''})"
    )

    total_groups = live_groups + dead_groups
    if not all_data:
        print("  [skip] every group is degenerate")
        wandb.log(
            {
                "step": step + 1,
                "mean_reward": mean(prompt_mean_rewards),
                "mean_advantage": 0.0,
                "live_groups": 0,
                "dead_groups": dead_groups,
                "frac_degenerate": (dead_groups / total_groups) if total_groups else 1.0,
                "batches_drawn": batches_drawn,
                "skipped": 1,
            }
        )
        continue

    train_start = time.time()

    # Microbatch the training side: split the rollouts into chunks of
    # MICROBATCH_SIZE, accumulate gradients across them, then one optim_step.
    # `zero_grad=True` only on the first chunk — the rest add their gradients
    # to the running accumulation. This is mathematically the same gradient
    # as one big forward_backward over `all_data`, just at lower peak memory.
    all_losses: list[float] = []
    sum_gradient: dict[str, float] = {}
    num_chunks = (len(all_data) + MICROBATCH_SIZE - 1) // MICROBATCH_SIZE
    for chunk_idx in range(num_chunks):
        chunk = all_data[chunk_idx * MICROBATCH_SIZE : (chunk_idx + 1) * MICROBATCH_SIZE]
        fb_result = training_client.forward_backward(
            data=chunk,
            loss_fn=LOSS_FN,
            loss_fn_config=LOSS_FN_CONFIG,
            zero_grad=(chunk_idx == 0),
            return_logprobs=False,
        ).result()
        if fb_result.loss is not None:
            chunk_losses = (
                [fb_result.loss] if isinstance(fb_result.loss, float) else fb_result.loss
            )
            all_losses.extend(chunk_losses)
        if fb_result.sum_gradient:
            sum_gradient = fb_result.sum_gradient  # last chunk wins; informational only

    training_client.optim_step(optimizer_params=OPTIMIZER_PARAMS).result()
    train_secs = time.time() - train_start

    loss_val = mean(all_losses)
    mean_reward = mean(prompt_mean_rewards)
    first_batch_pass = mean(first_batch_rewards)
    truncated_frac = truncated_count / total_rollouts if total_rollouts else 0.0
    print(
        f"  Loss: {loss_val:.4f}  Mean reward: {mean_reward:.2f}  "
        f"First-batch pass: {first_batch_pass:.2f}  "
        f"Truncated: {truncated_frac:.0%}  "
        f"(sample={sample_secs:.1f}s, train={train_secs:.1f}s, "
        f"microbatches={num_chunks})"
    )

    log_dict = {
        "step": step + 1,
        "loss": loss_val,
        "mean_reward": mean_reward,
        "first_batch_pass": first_batch_pass,
        "truncated_frac": truncated_frac,
        "mean_advantage": mean(all_advantages),
        "live_groups": live_groups,
        "dead_groups": dead_groups,
        "frac_degenerate": (dead_groups / total_groups) if total_groups else 0.0,
        "live_group_frac": (live_groups / total_groups) if total_groups else 0.0,
        "rollouts_used": len(all_data),
        "batches_drawn": batches_drawn,
        "sample_secs": sample_secs,
        "train_secs": train_secs,
    }
    if sum_gradient:
        for k, v in sum_gradient.items():
            log_dict[f"grad/{k}"] = v
    wandb.log(log_dict)

    # Sync new weights to a fresh sampler for the next step. Drop the previous
    # step's checkpoint dir so we don't leak per-step snapshots.
    step_ckpt = f"{CHECKPOINT_ROOT}-step{step}"
    sampling_client = training_client.save_weights_and_get_sampling_client(
        checkpoint_path=step_ckpt,
        tp_size=SAMPLING_TP_SIZE,
        engine_kwargs={
            "disable_radix_cache": True,
        },
    )
    sampling_client.wait_until_ready()
    if prev_step_ckpt and os.path.isdir(prev_step_ckpt):
        shutil.rmtree(prev_step_ckpt, ignore_errors=True)
    prev_step_ckpt = step_ckpt

print("\n" + "=" * 60)
print("GSM8K GRPO Training Complete!")
print("=" * 60)

wandb.finish()
