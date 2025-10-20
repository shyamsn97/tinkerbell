# Tensor Parallelism Example

This directory contains a complete example of using PyTorch's latest tensor parallelism APIs with transformer models.

## Files

- **`test_tensor_parallelism.py`**: Main example showing tensor parallelism with `parallelize_module`
- **`verify_tensor_parallel_setup.py`**: Helper script to verify your environment is set up correctly

## What is Tensor Parallelism?

Tensor parallelism splits individual layers of a model across multiple GPUs. Unlike data parallelism (where each GPU has a full copy of the model), tensor parallelism shards the **weight matrices** themselves across GPUs.

For example, in a transformer's MLP layer:
- GPU 0 holds columns 0-255 of the weight matrix
- GPU 1 holds columns 256-511 of the weight matrix

This enables training models that are too large to fit on a single GPU.

## Key APIs Used

### `torch.distributed.tensor.parallel.parallelize_module`
Main function to apply tensor parallelism to a model.

### `torch.distributed.tensor.parallel.ColwiseParallel`
Splits weight matrices column-wise. Used for layers that expand dimensions (like the first MLP layer).

### `torch.distributed.tensor.parallel.RowwiseParallel`
Splits weight matrices row-wise. Used for layers that reduce dimensions (like the second MLP layer).

### `torch.distributed.device_mesh.init_device_mesh`
Creates a device mesh that defines how GPUs are organized for parallelism.

## Requirements

```bash
# PyTorch with CUDA (>=2.3.0 for tensor parallel support)
pip install torch>=2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Transformers
pip install transformers

# Multiple GPUs (can run on 1 GPU but designed for multi-GPU)
```

## Usage

### 1. Verify Your Setup

```bash
python verify_tensor_parallel_setup.py
```

This will check:
- PyTorch version (needs >= 2.3.0)
- CUDA availability
- Number of GPUs
- Required modules

### 2. Run the Example

**With 2 GPUs:**
```bash
torchrun --nproc_per_node=2 test_tensor_parallelism.py
```

**With 4 GPUs:**
```bash
torchrun --nproc_per_node=4 test_tensor_parallelism.py
```

**With specific GPUs:**
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_tensor_parallelism.py
```

**Single GPU (for testing):**
```bash
torchrun --nproc_per_node=1 test_tensor_parallelism.py
```

## What the Example Does

1. **Initializes distributed environment** using NCCL backend
2. **Creates a device mesh** across all available GPUs
3. **Loads a GPT-2 model** (6 layers, 512 hidden size for demo)
4. **Applies tensor parallelism** to all MLP and attention layers using `parallelize_module`
5. **Performs one training step**:
   - Forward pass
   - Loss computation
   - Backward pass
   - Gradient clipping
   - Optimizer update
6. **Reports results** from rank 0

## Understanding the Parallelization Strategy

The example parallelizes transformer layers as follows:

```python
# For each transformer layer i:
parallelize_plan = {
    # MLP layers (main compute bottleneck)
    f"transformer.h.{i}.mlp.c_fc": ColwiseParallel(),      # Feed-forward expansion
    f"transformer.h.{i}.mlp.c_proj": RowwiseParallel(),    # Feed-forward projection
    
    # Attention layers
    f"transformer.h.{i}.attn.c_attn": ColwiseParallel(),   # Attention QKV projection
    f"transformer.h.{i}.attn.c_proj": RowwiseParallel(),   # Attention output
}
```

### Why ColwiseParallel then RowwiseParallel?

In a typical MLP layer: `output = W2 @ gelu(W1 @ input)`

- `W1` (c_fc): Expands from hidden_size → 4*hidden_size → **ColwiseParallel**
  - Each GPU computes a portion of the intermediate activations
  - No communication needed during forward pass
  
- `W2` (c_proj): Projects from 4*hidden_size → hidden_size → **RowwiseParallel**
  - Each GPU computes partial sums
  - AllReduce needed to combine results

This pattern minimizes communication between GPUs.

## Differences from Data Parallelism (DDP)

| Feature | Data Parallelism (DDP) | Tensor Parallelism |
|---------|------------------------|-------------------|
| Model copy | Full copy on each GPU | Model split across GPUs |
| Memory per GPU | Full model size | Model size / num_GPUs |
| Communication | Gradients after backward | Activations during forward/backward |
| Use case | Model fits on 1 GPU | Model too large for 1 GPU |
| Scaling | Good for large batches | Good for large models |

## Expected Output

```
======================================================================
PyTorch Tensor Parallelism with parallelize_module
======================================================================
World size (number of GPUs): 2
Global rank: 0
Local rank: 0
PyTorch version: 2.3.0
CUDA available: True
CUDA device count: 2

======================================================================
Loading Model with Tensor Parallelism
======================================================================
Model: gpt2
Layers: 6
Hidden size: 512
Attention heads: 8

Parallelizing 24 layers...
...

======================================================================
Training Step
======================================================================
Batch size: 4
Sequence length: 32

[1/4] Forward pass...
      Loss: 10.234567

[2/4] Backward pass (computing gradients)...

[3/4] Gradient clipping...
      Gradient norm: 12.345678

[4/4] Optimizer step (updating weights)...

======================================================================
Step Complete! Final Loss: 10.234567
======================================================================

✓ Tensor parallelism example completed successfully!
  - Model was sharded across 2 GPUs
  - Each GPU held a portion of each weight matrix
  - Training step completed with loss: 10.234567
```

## Troubleshooting

### ImportError: cannot import name 'parallelize_module'
Your PyTorch version is too old. Upgrade to >= 2.3.0:
```bash
pip install --upgrade torch>=2.3.0
```

### NCCL Error
Make sure all GPUs can communicate. Check:
```bash
nvidia-smi topo -m  # View GPU topology
```

### CUDA Out of Memory
- Reduce batch size in the example
- Use fewer/smaller layers
- Use gradient checkpointing (not shown in basic example)

### Hanging at Initialization
- Check that `torchrun` is setting up environment variables correctly
- Verify all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
- Check firewall settings (for multi-node setups)

## Advanced Usage

### Combining with Other Parallelism Strategies

Tensor parallelism can be combined with:
- **Data Parallelism**: Use a 2D device mesh
- **Pipeline Parallelism**: Split layers across different GPU sets
- **Sequence Parallelism**: Split sequence dimension

Example 2D mesh (4 GPUs: 2 tensor parallel, 2 data parallel):
```python
device_mesh = init_device_mesh("cuda", (2, 2))  # (TP, DP)
```

### Using with Larger Models

For models like LLaMA, Mistral, or custom models:

1. Identify the layer names (use `print(model)`)
2. Create parallelization plan for all linear layers
3. Consider using `SequenceParallel` for even more memory savings

### Memory Savings

With tensor parallelism across N GPUs:
- **Model parameters**: ~1/N of original (each GPU holds a portion)
- **Activations**: Can also be reduced with sequence parallelism
- **Communication overhead**: Increases with more GPUs

## References

- [PyTorch Distributed Tensor Parallel Docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [PyTorch TP Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Original tensor parallelism research

## License

Same as parent project.

