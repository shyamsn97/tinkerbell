# Tinkerbell

A small open-source reimplementation of [Tinker](https://tinker-docs.thinkingmachines.ai/) from Thinking Machines.

Tinkerbell is a thin distributed training and inference library for LLMs on top of Ray, PyTorch, and [SGLang](https://github.com/sgl-project/sglang). You write a normal Python training loop on your laptop with a handful of primitives (`forward_backward`, `optim_step`, `sample`, …) and Tinkerbell runs it across the GPUs of a remote server.

The API surface is intentionally small:

| Primitive | What it does |
| --- | --- |
| `ServiceClient` | HTTP client that talks to the server |
| `create_training_client` | Spin up a tensor-parallel training group for a base model |
| `create_sampling_client` | Spin up an SGLang inference group |
| `forward_backward` / `optim_step` / `zero_grad` | One step of training |
| `save_weights_and_get_sampling_client` | Snapshot adapter weights and get a sampler that uses them |
| `sample` / `sample_many` | Generate from the current sampler |

Everything runs through one process on the server side (Ray + Ray Serve), so state stays consistent and the same checkpoint can be trained and served without moving files around.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install peft               # LoRA
pip install "sglang[all]"      # high-throughput inference
pip install modal              # remote GPU deployment
```

Requirements: Python 3.9+, PyTorch 2.0+, Ray 2.0+, Transformers.

## Deployment

Tinkerbell needs a server to run training and sampling actors. There are two supported modes:

**Local** (single host, Ray runs in-process):

```python
from tinkerbell.client import ServiceClient

service = ServiceClient.deploy()  # serves at http://127.0.0.1:8000
```

**Modal** (remote GPUs, recommended):

```python
from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig

deploy_config = ModalDeployConfig(
    gpu="H100",
    num_gpus=2,
    timeout=86400,
    scaledown_window=600,
)

service = ServiceClient.deploy_or_connect(deploy_config)
```

`deploy_or_connect` reuses a running Modal deployment if there is one, otherwise it deploys a new one.

## Quickstart: full SFT loop

Train a 0.6B model on a single chat example. End to end, including sampling from the trained adapter.

```python
from tinker.types import LoraConfig
from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig
from tinkerbell.renderer import Renderer, TrainOnWhat

service = ServiceClient.deploy_or_connect(ModalDeployConfig(gpu="H100", num_gpus=1))

# 1. Training group
training_client = service.create_training_client(
    base_model="Qwen/Qwen3-0.6B-Base",
    model_name="qwen-sft",
    tp_size=1,
    lora_config=LoraConfig(rank=64, train_attn=True, train_mlp=True).model_dump(),
    model_kwargs={"torch_dtype": "bfloat16", "gradient_checkpointing": True},
)
training_client.wait_until_ready()

# 2. Build training data with the renderer (handles chat template, label masking, shifting)
conversations = [
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ],
]
data = training_client.build_chat_samples(
    messages=conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
)

# 3. Train
for step in range(50):
    fb = training_client.forward_backward(data=data, loss_fn="cross_entropy").result()
    training_client.optim_step(
        optimizer_params={"name": "adamw", "lr": 1e-4, "weight_decay": 0.0}
    ).result()
    print(f"step {step}  loss={fb.loss}")

# 4. Snapshot weights and get a sampler that uses them
sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/tmp/qwen-sft-step50",
    tp_size=1,
)
sampling_client.wait_until_ready()

# 5. Generate
prompt_ids = training_client.build_chat_samples(
    messages=[[{"role": "user", "content": "What is the capital of France?"}]],
    include_labels=False,
)[0].model_input.to_ints()

out = sampling_client.sample(
    input_ids=prompt_ids,
    sampling_params={"max_new_tokens": 32, "temperature": 0.7},
).result()
print(out.output)
```

Both `forward_backward` and `sample` return futures — call `.result()` to block, or fire off many in parallel and join later.

## Async / batched sampling

For RL or eval workloads where you want N completions from M prompts in one round trip:

```python
batch_kwargs = [
    {"input_ids": ids, "sampling_params": {"max_new_tokens": 256, "temperature": 1.0}}
    for ids in prompts
    for _ in range(8)            # 8 samples per prompt
]
samples = sampling_client.sample_many(batch_kwargs).result()  # flat list
```

Set `sampling_params={"return_logprob": True, "top_logprobs_num": 1, ...}` if you need per-token sampler logprobs (required for importance-sampling losses, GRPO etc.).

## RL example: GRPO on GSM8K

`scripts/grpo_example.py` is a complete, ~450-line GRPO loop that trains Qwen3-1.7B on GSM8K. The training-relevant core looks like this:

```python
for step in range(NUM_GRPO_STEPS):
    # 1. Sample N completions per prompt
    samples = sampling_client.sample_many([
        {"input_ids": ids, "sampling_params": SAMPLING_PARAMS}
        for ids in prompt_token_ids for _ in range(NUM_SAMPLES_PER_PROMPT)
    ]).result()

    # 2. Score, group-relative advantages
    rewards = [reward_fn(s.output, gt) for s, gt in zip(samples, gts)]
    advantages = [r - mean(group) for group, r in groups(rewards)]

    # 3. Build a Datum per rollout (prompt + sampled tokens + sampler logprobs + advantage)
    data = [build_grpo_datum(...) for sample, adv in zip(samples, advantages)]

    # 4. Microbatched forward_backward + one optim_step
    for chunk in chunked(data, MICROBATCH_SIZE):
        training_client.forward_backward(
            data=chunk,
            loss_fn="importance_sampling",
        ).result()
    training_client.optim_step(optimizer_params=OPTIMIZER_PARAMS).result()

    # 5. Sync new weights to a fresh sampler for the next step
    sampling_client = training_client.save_weights_and_get_sampling_client(
        checkpoint_path=f"/tmp/grpo-step{step}", tp_size=1,
    )
    sampling_client.wait_until_ready()
```

The full script also handles dynamic oversampling (keep drawing prompts until enough live groups), microbatched gradient accumulation, W&B logging, and per-step checkpoint cleanup.

## Renderer

The `Renderer` (and its higher-level wrapper `training_client.build_chat_samples`) does three things you'd otherwise have to do by hand:

1. Apply the model's chat template.
2. Mask labels: by default only the last assistant message contributes to the loss; everything else is `-100`.
3. Right-shift labels so position `i` predicts token `i+1`.

`TrainOnWhat` controls what counts as a labeled span:

```python
TrainOnWhat.ALL_ASSISTANT_MESSAGES   # train on every assistant turn
TrainOnWhat.LAST_ASSISTANT_MESSAGE   # train only on the final assistant turn
TrainOnWhat.NOTHING                  # inference-only (used by sampling)
```

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│  Tinkerbell Server  (FastAPI + Ray Serve, single replica)│
│                                                          │
│   /forward_backward, /optim_step, /sample, /sample_batch │
│              │                          │                │
│              ▼                          ▼                │
│   ┌────────────────────┐    ┌────────────────────┐       │
│   │   TrainGroup       │    │   Sampler          │       │
│   │  Ray actors, TP    │    │  SGLang engine     │       │
│   │  PyTorch + LoRA    │    │  LoRA hot-swap     │       │
│   └────────────────────┘    └────────────────────┘       │
└──────────────────────────────────────────────────────────┘
                  ▲ HTTP submit + poll
                  │
              ServiceClient  (your laptop)
```

- One server process owns all Ray actors. The HTTP layer is a thin gateway: every "heavy" call returns a `job_id` immediately and the client polls `/poll`. Job state lives in Ray's internal KV store, so it survives across HTTP replicas without losing track of in-flight work.
- Training and sampling run as separate Ray actor groups. They share the same model weights via on-disk LoRA checkpoints and SGLang's hot-swap path, which is why `save_weights_and_get_sampling_client` can hand you a sampler with the freshly-trained adapter without restarting anything heavy.

## Repo layout

```text
tinkerbell/
  api/         # FastAPI + Ray Serve gateway, Modal deploy helper
  client/      # ServiceClient, TrainingClient, SamplingClient
  runtime/     # Ray session, futures, resource specs
  training/    # TrainGroup, Trainer, loss functions
  sampling/    # Sampler, SGLang actor
  renderer.py  # Chat-template + label-masking helper
  types/       # Pydantic request/response models, deploy configs

scripts/
  chat_sft.py        # SFT against an OpenAssistant-style chat dataset
  grpo_example.py    # GRPO on GSM8K, end-to-end RL loop
  multi_gpu.py       # Tensor-parallel sanity check
  test_sampling.py   # Sampling-only smoke test
  count_lines.py     # Library line count (excludes scripts/, examples/, tests/)
```

## License

See `LICENSE`.
