# Tinkerbell

An open-source implementation of [Tinker](https://tinker-docs.thinkingmachines.ai/) from Thinking Machines.

Tinkerbell is a distributed training and inference framework for large language models, built on Ray and PyTorch. Like Tinker, it provides a simple API that lets you focus on your data and loss functions while handling the complexity of distributed training. You write a training loop that runs on your machine, and Tinkerbell figures out how to efficiently execute it across multiple GPUs.

**Key Philosophy** (inspired by Tinker):
- 📊 **You focus on**: Your datasets, loss functions, and training logic
- 💻 **You write**: Simple Python scripts with API calls like `forward_backward()`, `optim_step()`, `sample()`
- ⚡ **We handle**: Distributed training across GPUs, tensor parallelism, and infrastructure complexity

## Features

- 🚀 **Distributed Training**: Multi-GPU training with tensor parallelism
- 🎯 **LoRA Support**: Parameter-efficient fine-tuning with Low-Rank Adaptation
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

### 2. Full Fine-Tuning Example

```python
from tinkerbell.client.service import ServiceClient
from tinkerbell.types import Datum, ModelInput, TensorData

# Initialize service client and create training actors
service = ServiceClient(server_url="http://localhost:8000")
client = service.create_training_client(
    model_id="meta-llama/Llama-3.2-1B",
    tp_size=2,  # Number of GPUs
    model_kwargs={"torch_dtype": "bfloat16"},
    parallelize_plan={
        "model.layers.*.self_attn.q_proj": "column",
        "model.layers.*.self_attn.k_proj": "column",
        "model.layers.*.self_attn.v_proj": "column",
        "model.layers.*.self_attn.o_proj": "row",
    },
    wait_until_ready=True,
)

# Prepare training data
prompt = "Question: What is the capital of France? Answer:"
completion = " Paris"
full_text = prompt + completion

# Tokenize
tokenized = client.tokenizer([full_text])
input_ids = tokenized["input_ids"][0]

# Create labels
prompt_tokens = client.tokenizer([prompt])["input_ids"][0]
labels = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens):]

datum = Datum(
    model_input=ModelInput(tokens=input_ids),
    loss_fn_inputs={"labels": TensorData.from_list(labels)},
)

# Training loop
for step in range(100):
    client.zero_grad()
    result = client.forward_backward(data=[datum])
    loss_info = client.get_result(result.request_id)
    print(f"Step {step}, Loss: {loss_info['loss']}")
    client.optim_step(optimizer_params={"name": "adamw", "lr": 1e-4})

# Save checkpoint
client.save_checkpoint("/tmp/my_model")
```

### 3. LoRA Fine-Tuning Example

```python
from tinkerbell.client.service import ServiceClient
from tinkerbell.types import LoraConfig

# Create LoRA configuration
lora_config = LoraConfig(
    rank=8,              # LoRA rank
    train_attn=True,     # Apply LoRA to attention layers
    train_mlp=True,      # Apply LoRA to MLP layers
    train_unembed=True,  # Apply LoRA to output layer
)

# Create training actors with LoRA
service = ServiceClient(server_url="http://localhost:8000")
client = service.create_training_client(
    model_id="meta-llama/Llama-3.2-1B",
    tp_size=1,
    model_kwargs={"torch_dtype": "bfloat16"},
    lora_config=lora_config.model_dump(),
    wait_until_ready=True,
)

# Training works the same way as full fine-tuning
# ... (same training loop as above)

# Save LoRA adapters (much smaller than full model!)
client.save_checkpoint("/tmp/lora_adapters")
```

### 4. Inference After Training

```python
# Save checkpoint and create sampling actor in one call
sampling_client = client.save_weights_and_get_sampling_client(
    checkpoint_path="/tmp/my_model",
    tp_size=1,
    wait_until_ready=True,
)

# Generate text
response = sampling_client.sample(
    text="Question: What is the capital of France? Answer:",
    sampling_params={"max_new_tokens": 50, "temperature": 0.7},
)

print(response.outputs[0])
```

## LoRA Configuration Options

```python
LoraConfig(
    rank=8,              # LoRA rank (higher = more parameters)
    seed=42,             # Optional: for reproducible initialization
    train_attn=True,     # Apply to attention layers (Q,K,V,O)
    train_mlp=True,      # Apply to MLP/FFN layers
    train_unembed=True,  # Apply to output embedding layer
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

See the `examples/` directory for more:
- `lora_training_example.py` - Complete LoRA fine-tuning example
- `training_client_example.py` - Full fine-tuning example
- `tensor_parallel_minimal.py` - Multi-GPU training setup

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Ray 2.0+
- Transformers
- PEFT (for LoRA)
- SGLang (for inference)

## License

See LICENSE file for details.
