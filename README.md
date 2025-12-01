# Tinkerbell

An small(ish) open-source reimplementation of [Tinker](https://tinker-docs.thinkingmachines.ai/) from Thinking Machines.

Tinkerbell is a distributed training and inference framework for large language models, built on Ray, PyTorch, and [SGlang](https://github.com/sgl-project/sglang). Like Tinker, it provides a simple API that lets you focus on your data and loss functions while handling the complexity of distributed training. You write a training loop that runs on your machine, and Tinkerbell figures out how to efficiently execute it across multiple GPUs.

**Key Philosophy** (inspired by Tinker):
- 📊 **You focus on**: Your datasets, loss functions, and training logic
- 💻 **You write**: Simple Python scripts with API calls like `forward_backward()`, `optim_step()`, `sample()`
- ⚡ **We handle**: Distributed training across GPUs, tensor parallelism, and infrastructure complexity

## Features

- 🚀 **Distributed Training**: Multi-GPU training with tensor parallelism
- 🎯 **LoRA Support**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- 📝 **Smart Renderer**: Automatic chat formatting, label masking, and label shifting
- ⚡ **Fast Inference**: Integrated SGLang backend for high-performance sampling
- 🔄 **Seamless Workflow**: Train → Save → Load → Inference in one API
- 📦 **Ray-Powered**: Built on Ray for distributed computing

## Installation

```bash
pip install -e .
```

### Optional Dependencies

```bash
# For LoRA support
pip install peft

# For inference (SGLang)
pip install "sglang[all]"
```

## Quick Start

### 1. Start the Service

```python
from tinkerbell.service.server import deploy_service

# Deploy the training service
server_url = deploy_service(
    server_url="http://localhost:8000",
    max_wait_time=600.0,
    clock_cycle=10.0,
)
```

### 2. Training with Renderer (Recommended)

```python
from tinkerbell.client import ServiceClient
from tinkerbell.renderer import Renderer, TrainOnWhat

# Initialize service client and create training actors
service = ServiceClient(server_url="http://localhost:8000")
training_client = service.create_training_client(
    model_id="meta-llama/Llama-3.2-1B",
    tp_size=2,  # Number of GPUs
    model_kwargs={"torch_dtype": "bfloat16"},
    parallelize_plan={
        "model.layers.*.self_attn.q_proj": "column",
        "model.layers.*.self_attn.k_proj": "column",
        "model.layers.*.self_attn.v_proj": "column",
        "model.layers.*.self_attn.o_proj": "row",
    },
)
training_client.wait_until_ready()

# Prepare training conversations
conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
]

# Use Renderer to create properly formatted training data
tokenizer = training_client.get_tokenizer()
renderer = Renderer(tokenizer)

training_data = renderer.build_chat_examples(
    conversations=conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,  # Only train on assistant's response
    mask_value=-100,
)
# Returns a list of Datum objects, each with structure:
# Datum(
#     model_input=ModelInput(input_ids=[...], attention_mask=[...]),
#     loss_fn_inputs={"labels": TensorData([...])}  # -100 for masked tokens
# )

# Training loop
for step in range(100):
    training_client.zero_grad()
    
    response = training_client.forward_backward(
        data=training_data,
        forward_kwargs={},
    )
    result = response.result()
    
    losses = result.loss
    if losses:
        avg_loss = sum(losses) / len(losses)
        print(f"Step {step}, Loss: {avg_loss:.4f}")
    
    training_client.optim_step(
        optimizer_params={
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    ).result()

# Save checkpoint
training_client.save_checkpoint("/tmp/my_model")
```

### 3. LoRA Fine-Tuning Example

```python
from tinkerbell.client import ServiceClient
from tinkerbell.types import LoraConfig
from tinkerbell.renderer import Renderer, TrainOnWhat

# Create LoRA configuration
lora_config = LoraConfig(
    rank=8,              # LoRA rank
    seed=42,             # For reproducible initialization
    train_attn=True,     # Apply LoRA to attention layers
    train_mlp=True,      # Apply LoRA to MLP layers
    train_unembed=False, # Apply LoRA to output layer
)

# Create training actors with LoRA
service = ServiceClient(server_url="http://localhost:8000")
training_client = service.create_training_client(
    model_id="meta-llama/Llama-3.2-1B",
    tp_size=1,
    lora_config=lora_config.model_dump(),  # Enable LoRA
    model_kwargs={"torch_dtype": "bfloat16"},
)
training_client.wait_until_ready()

# Prepare training data with Renderer
tokenizer = training_client.get_tokenizer()
renderer = Renderer(tokenizer)
conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ],
]
training_data = renderer.build_chat_examples(
    conversations=conversations,
    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    mask_value=-100,
)

# Training loop (same as above, but only LoRA parameters are updated!)
for step in range(100):
    training_client.zero_grad()
    response = training_client.forward_backward(data=training_data, forward_kwargs={})
    result = response.result()
    training_client.optim_step(
        optimizer_params={"name": "adamw", "lr": 1e-4, "weight_decay": 0.01}
    ).result()

# Save LoRA adapters (much smaller than full model!)
training_client.save_checkpoint("/tmp/lora_adapters")
```

### 4. Inference After Training

```python
# Save checkpoint and create sampling actor in one call
sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/tmp/my_model",
    tp_size=1,
    wait_until_ready=True,
)

# Prepare inference prompts
inference_prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
]

# Tokenize prompts for inference
formatted_prompts = tokenizer.apply_chat_template(
    inference_prompts, 
    add_generation_prompt=True, 
    tokenize=False
)
encoded = tokenizer(formatted_prompts, padding=True, return_tensors="pt")

# Generate text
tinkerbell_future = sampling_client.sample(
    input_ids=encoded["input_ids"][0],
    sampling_params={"max_new_tokens": 100, "temperature": 0.7},
)
outputs = tinkerbell_future.result()

print(outputs)
```

## LoRA Configuration Options

```python
LoraConfig(
    rank=8,              # LoRA rank (higher = more parameters)
    seed=42,             # Optional: for reproducible initialization
    train_attn=True,     # Apply to attention layers (Q,K,V,O)
    train_mlp=True,      # Apply to MLP/FFN layers
    train_unembed=False, # Apply to output embedding layer
)
```

**Benefits of LoRA:**
- 💾 **Memory Efficient**: Only trains 0.1-1% of parameters
- ⚡ **Faster Training**: Fewer parameters = faster updates
- 💰 **Lower Storage**: Adapter files are 10-100MB vs full model GBs
- 🎯 **Same Quality**: Often matches full fine-tuning performance

## Architecture

```
┌────────────────────────────────────────────┐
│        Tinkerbell Service (Ray Serve)      │
│  ┌─────────────────────────────────────┐   │
│  │      Training Manager               │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │   Training Actor (Rank 0)   │    │   │
│  │  │   - PyTorch Model           │    │   │
│  │  │   - Tensor Parallel         │    │   │
│  │  │   - LoRA (optional)         │    │   │
│  │  └─────────────────────────────┘    |   |
|  |              ...                    |   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │   Training Actor (Rank n)   │    │   │
│  │  └─────────────────────────────┘    │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │      Sampling Manager               │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │   Sampling Actor 0 (SGLang) │    │   │
│  │  └─────────────────────────────┘    |   |
|  |                ...                  |   |
│  |  ┌─-───────────────────────────┐    │   │
│  │  │   Sampling Actor n (SGLang) │    │   │
│  │  └─────────────────────────────┘    │   │
│  └─────────────────────────────────────┘   │
└────────────────────────────────────────────┘
```

## Examples

See the `examples/` and `scripts/` directories for more:
- `scripts/test_client.py` - Complete training and inference example with LoRA and Renderer
- `examples/tutorial.py` - Getting started tutorial
- `examples/futures_api_example.py` - Async API examples

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Ray 2.0+
- Transformers
- PEFT (for LoRA)
- SGLang (for inference)

## License

See LICENSE file for details.
