"""GSM8K GRPO using the Tinkerbell server.

Sibling of `grpo_tinker.py` — same dataset, prompt format, reward function,
hyperparameters, and W&B logging shape. The only differences are mechanical:

* Tinker's hosted `ServiceClient()` becomes Tinkerbell's
  `ServiceClient.deploy_or_connect(...)`.
* The cookbook's datum format (`target_tokens`, `logprobs`, `advantages`) is
  accepted directly; the Tinkerbell client translates those keys at the wire
  boundary.

Use this to A/B against `grpo_tinker.py` for an apples-to-apples comparison
of Tinkerbell vs hosted Tinker.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import cast

import torch
import wandb
from datasets import DatasetDict, load_dataset
from tinkerbell.renderer import get_text_content

import tinkerbell as tinker
from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig

# ---------------------------------------------------------------------------
# Config (kept aligned with scripts/grpo/grpo_tinker.py for direct A/B)
# ---------------------------------------------------------------------------

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-8B-Base")
GPU_TYPE = os.getenv("GPU_TYPE", "H100")
NUM_GPUS = int(os.getenv("NUM_GPUS", "3"))
TRAIN_TP_SIZE = int(os.getenv("TRAIN_TP_SIZE", "2"))
SAMPLING_TP_SIZE = int(os.getenv("SAMPLING_TP_SIZE", "1"))
ADAPTER_NAME = "grpo-adapter"
CHECKPOINT_ROOT = "/tmp/grpo-checkpoint"

LORA_RANK = int(os.getenv("LORA_RANK", "32"))

NUM_GSM8K_EXAMPLES = int(os.getenv("NUM_GSM8K_EXAMPLES", "16"))
NUM_SAMPLES_PER_PROMPT = int(os.getenv("NUM_SAMPLES_PER_PROMPT", "8"))
NUM_GRPO_STEPS = int(os.getenv("NUM_GRPO_STEPS", "100"))
TRAIN_MICROBATCH_SIZE = int(os.getenv("TRAIN_MICROBATCH_SIZE", "16"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "4e-5"))
BETA1 = float(os.getenv("BETA1", "0.9"))
BETA2 = float(os.getenv("BETA2", "0.95"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))


# ---------------------------------------------------------------------------
# Reward / advantage helpers (identical to grpo_tinker.py)
# ---------------------------------------------------------------------------


def extract_gsm8k_final_answer(text: str) -> str:
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
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def normalize_number(ans: str) -> str:
    ans = ans.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"-?\d+\.?\d*", ans)
    return match.group() if match else ans.lower()


def compute_gsm8k_reward(output_text: str, ground_truth: str) -> float:
    answer = extract_boxed(output_text)
    if answer is None:
        return 0.0
    return 1.0 if normalize_number(answer) == normalize_number(ground_truth) else 0.0


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


QUESTION_SUFFIX = (
    " Provide a numerical answer without units, written inside \\boxed{}."
)
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


# ---------------------------------------------------------------------------
# Main loop (mirrors grpo_tinker.main)
# ---------------------------------------------------------------------------


async def main() -> None:
    ds = cast(DatasetDict, load_dataset("openai/gsm8k", name="main"))
    train_data = ds["train"]

    deploy_config = ModalDeployConfig(
        gpu=GPU_TYPE,
        num_gpus=NUM_GPUS,
        timeout=86400,
        scaledown_window=600,
        max_inputs=200,
        max_wait_time=1200.0,
    )
    service_client = ServiceClient.deploy_or_connect(deploy_config, redeploy=True)

    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        tp_size=TRAIN_TP_SIZE,
        model_name="qwen3-grpo",
        adapter_name=ADAPTER_NAME,
        initialize_base_model=False,
        model_kwargs={
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
    )
    renderer = training_client.get_renderer()

    sampling_params = tinker.SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker.AdamParams(
        learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2
    )

    wandb.init(
        project="tinkerbell-grpo",
        name=f"gsm8k-grpo-tinkerbell-{BASE_MODEL.split('/')[-1]}",
        settings=wandb.Settings(x_save_requirements=False, disable_code=True),
        config={
            "backend": "tinkerbell",
            "base_model": BASE_MODEL,
            "renderer": "tinkerbell",
            "lora_rank": LORA_RANK,
            "num_gsm8k_examples": NUM_GSM8K_EXAMPLES,
            "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
            "num_grpo_steps": NUM_GRPO_STEPS,
            "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
            "loss_fn": "importance_sampling",
            "learning_rate": LEARNING_RATE,
            "betas": [BETA1, BETA2],
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
    )

    for step in range(NUM_GRPO_STEPS):
        print(f"\n--- GRPO Step {step + 1}/{NUM_GRPO_STEPS} ---")

        # Slice today's batch off the dataset (cookbook walks the dataset).
        batch_start = (step * NUM_GSM8K_EXAMPLES) % len(train_data)
        batch_indices = [
            (batch_start + i) % len(train_data) for i in range(NUM_GSM8K_EXAMPLES)
        ]
        batch_rows = train_data.select(batch_indices)
        print(f"  Batch indices {batch_indices[:3]}...{batch_indices[-3:]}")

        # 1. Sync sampler with the current policy weights.
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            checkpoint_path=f"{CHECKPOINT_ROOT}-step{step}",
            tp_size=SAMPLING_TP_SIZE,
            engine_kwargs={"disable_radix_cache": True},
        )

        # 2. Submit all groups concurrently.
        prompts_P: list[tinker.ModelInput] = []
        coros = []
        for question in batch_rows["question"]:
            convo = [
                *FEWSHOT_PREFIX,
                {"role": "user", "content": question + QUESTION_SUFFIX},
            ]
            prompt = renderer.build_generation_prompt(convo)
            prompts_P.append(prompt)
            coros.append(
                sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=NUM_SAMPLES_PER_PROMPT,
                    sampling_params=sampling_params,
                )
            )
        sample_results_P = await asyncio.gather(*coros)

        # 3. Grade, compute advantages, build datums.
        datums_D: list[tinker.Datum] = []
        prompt_mean_rewards: list[float] = []
        first_batch_rewards: list[float] = []
        all_advantages: list[float] = []
        live_groups = 0
        dead_groups = 0
        truncated_count = 0
        total_rollouts = 0

        for sample_result, prompt, answer_text in zip(
            sample_results_P, prompts_P, batch_rows["answer"]
        ):
            ground_truth = extract_gsm8k_final_answer(answer_text)

            rewards_G: list[float] = []
            tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []

            for sequence in sample_result.sequences:
                tokens_G_T.append(sequence.tokens)
                logprobs_G_T.append(sequence.logprobs)
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                content = get_text_content(parsed_message)
                rewards_G.append(compute_gsm8k_reward(content, ground_truth))
                total_rollouts += 1
                # Tinker exposes finish reason per sequence; "length" means
                # the model hit max_tokens before emitting a stop sequence.
                stop_reason = getattr(sequence, "stop_reason", None) or getattr(
                    sequence, "finish_reason", None
                )
                if stop_reason and "length" in str(stop_reason).lower():
                    truncated_count += 1

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            prompt_mean_rewards.append(mean_reward)
            first_batch_rewards.extend(rewards_G)

            if all(a == 0.0 for a in advantages_G):
                dead_groups += 1
                continue

            live_groups += 1
            print(f"  Prompt (gt={ground_truth})")
            print(f"    Rewards:    {[f'{r:.1f}' for r in rewards_G]}")
            print(f"    Advantages: {[f'{a:.3f}' for a in advantages_G]}")

            # Cookbook datum construction: prompt + tokens[:-1] as input,
            # target_tokens shifted with [0]*ob_len padding for the prompt
            # region, advantages zero on prompt and constant on action.
            ob_len = prompt.length - 1
            for tokens, logprobs, advantage in zip(
                tokens_G_T, logprobs_G_T, advantages_G
            ):
                model_input = prompt.append(
                    tinker.EncodedTextChunk(tokens=tokens[:-1])
                )
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = (
                    [0.0] * ob_len
                    + [advantage] * (model_input.length - ob_len)
                )

                datum = tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(target_tokens)
                        ),
                        "logprobs": tinker.TensorData.from_torch(
                            torch.tensor(padded_logprobs)
                        ),
                        "advantages": tinker.TensorData.from_torch(
                            torch.tensor(padded_advantages)
                        ),
                    },
                )
                datums_D.append(datum)
                all_advantages.append(advantage)

        # 4. Single forward_backward + optim_step (cookbook style).
        if datums_D:
            losses_for_step: list[float] = []
            for start in range(0, len(datums_D), TRAIN_MICROBATCH_SIZE):
                chunk = datums_D[start : start + TRAIN_MICROBATCH_SIZE]
                fwd_bwd_future = await training_client.forward_backward_async(
                    chunk, loss_fn="importance_sampling"
                )
                fwd_bwd_result = await fwd_bwd_future.result_async()
                if fwd_bwd_result.loss is not None:
                    losses = (
                        [fwd_bwd_result.loss]
                        if isinstance(fwd_bwd_result.loss, float)
                        else fwd_bwd_result.loss
                    )
                    losses_for_step.extend(losses)

            optim_future = await training_client.optim_step_async(adam_params)
            await optim_future.result_async()
            loss_val: float | None = mean(losses_for_step) if losses_for_step else None
        else:
            loss_val = None
            print("  [skip] every group is degenerate")

        total_groups = live_groups + dead_groups
        truncated_frac = truncated_count / total_rollouts if total_rollouts else 0.0
        mean_reward = mean(prompt_mean_rewards)
        first_batch_pass = mean(first_batch_rewards)

        print(
            f"  Loss: {loss_val if loss_val is None else f'{loss_val:.4f}'}  "
            f"Mean reward: {mean_reward:.2f}  "
            f"First-batch pass: {first_batch_pass:.2f}  "
            f"Truncated: {truncated_frac:.0%}  "
            f"Live: {live_groups}/{total_groups}"
        )

        wandb.log(
            {
                "step": step + 1,
                "loss": loss_val,
                "mean_reward": mean_reward,
                "first_batch_pass": first_batch_pass,
                "truncated_frac": truncated_frac,
                "mean_advantage": mean(all_advantages),
                "live_groups": live_groups,
                "dead_groups": dead_groups,
                "frac_degenerate": (
                    dead_groups / total_groups if total_groups else 0.0
                ),
                "live_group_frac": (
                    live_groups / total_groups if total_groups else 0.0
                ),
                "rollouts_used": len(datums_D),
                "train_microbatches": (
                    (len(datums_D) + TRAIN_MICROBATCH_SIZE - 1)
                    // TRAIN_MICROBATCH_SIZE
                    if datums_D
                    else 0
                ),
                "batches_drawn": 1,
            }
        )

    print("\n" + "=" * 60)
    print("GSM8K GRPO (Tinkerbell) Training Complete!")
    print("=" * 60)
    wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
