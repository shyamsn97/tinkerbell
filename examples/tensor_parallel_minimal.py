"""
Minimal Tensor Parallelism Example

This is a stripped-down version showing just the essential code for tensor parallelism.
For a full example with logging and error handling, see test_tensor_parallelism.py

Run with:
    modal run tensor_parallel_minimal.py
"""

import modal
import re
import fnmatch

def get_submodules_with_wildcard(model, pattern):
    """
    Get all submodules matching a wildcard pattern.

    Args:
        model: PyTorch model
        pattern: Pattern with wildcards (e.g., "model.layers.*.self_attn.q_proj")

    Returns:
        List of (name, module) tuples matching the pattern
    """
    # Convert wildcard pattern to regex
    regex_pattern = fnmatch.translate(pattern)
    regex = re.compile(regex_pattern)

    matching_modules = []
    for name, module in model.named_modules():
        if regex.match(name):
            matching_modules.append(name)

    return matching_modules

app = modal.App(name="tensor-parallel-minimal")

image = modal.Image.debian_slim().pip_install("torch", "torchvision", "torchaudio", "transformers")

def train_worker(rank, world_size):
    """Training function that runs on each GPU - must be at module level for pickling."""
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    def print_gpu_memory(prefix="", rank=0):
        """Print GPU memory usage for the current device."""
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(rank).total_memory / 1024**3

        print(f"[Rank {rank}] {prefix}")
        print(f"  GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB | Total: {total:.2f}GB")

    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:29500",
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)

    # Create device mesh with named dimensions
    device_mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("dp", "tp"))

    # Create model
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    config.n_layer = 4  # Small model for demo
    model = AutoModelForCausalLM.from_config(config)

    strategies = {
        "column": ColwiseParallel,
        "row": RowwiseParallel,
        "sequence": SequenceParallel,
    }

    # Define parallelization plan
    parallelize_plan = {
        # Note: Embedding layer is NOT parallelized - it remains replicated
        
        # Attention projections (all layers)
        "model.layers.*.self_attn.q_proj": "column",
        "model.layers.*.self_attn.k_proj": "column",
        "model.layers.*.self_attn.v_proj": "column",
        "model.layers.*.self_attn.o_proj": "row",

        # MLP projections (all layers)
        "model.layers.*.mlp.gate_proj": "column",
        "model.layers.*.mlp.up_proj": "column",
        "model.layers.*.mlp.down_proj": "row",
        
        # Note: lm_head is NOT parallelized to avoid issues with loss computation
        # "lm_head": "row",
    }

    module_parallelization_plan = {}

    for pattern in parallelize_plan.keys():
        strategy = strategies[parallelize_plan[pattern]]()
        module_names = get_submodules_with_wildcard(model, pattern)
        for name in module_names:
            module_parallelization_plan[name] = strategy

    # Use the TP dimension of the mesh
    model = parallelize_module(model, device_mesh["tp"], module_parallelization_plan)
    model = model.cuda()

    # Print
    print_gpu_memory(f"Model loaded on GPU {rank}", rank)

    # Setup optimizer - disable foreach to handle mixed DTensor/Tensor parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, foreach=False)

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(["Hello world!"], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].cuda()

    # Training step
    model.train()
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if rank == 0:
        print(f"✓ Training step complete! Loss: {loss.item():.4f}")

    # Cleanup
    dist.destroy_process_group()

@app.function(image=image, gpu="H100:4")
def train():
    import torch
    import torch.multiprocessing as mp

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Starting training with {world_size} GPUs")

    # Spawn processes - one per GPU
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


@app.local_entrypoint()
def main():
    train.remote()