"""
Tensor Parallelism Example with PyTorch's Latest APIs

This example demonstrates how to use torch.distributed.tensor.parallel with
parallelize_module to train a transformer model across multiple GPUs using
true tensor parallelism (not data parallelism).

Requirements:
- torch >= 2.3.0 (with tensor parallel support)
- transformers
- Multiple GPUs (works with 1 GPU but designed for multi-GPU)

Usage:
    # Multi-GPU with tensor parallelism (2 GPUs)
    torchrun --nproc_per_node=2 test_tensor_parallelism.py
    
    # Or with specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_tensor_parallelism.py
    
    # 4 GPUs
    torchrun --nproc_per_node=4 test_tensor_parallelism.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[Rank %(rank)s] %(message)s'
)


def setup_distributed():
    """Initialize the distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_parallelization_plan(model, num_layers=None):
    """
    Create a parallelization plan for GPT-2 style models.
    
    This function maps each MLP layer to use tensor parallelism:
    - c_fc (feed-forward expansion): ColwiseParallel
    - c_proj (feed-forward projection): RowwiseParallel
    
    For attention layers, we can also parallelize:
    - c_attn (attention projection): ColwiseParallel
    - c_proj (attention output): RowwiseParallel
    """
    parallelize_plan = {}
    
    # Determine the number of layers
    if num_layers is None:
        # Try to infer from model
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
        else:
            num_layers = 12  # Default for GPT-2
    
    # Parallelize each transformer layer
    for i in range(num_layers):
        # MLP layers - these are the main compute bottlenecks
        parallelize_plan[f"transformer.h.{i}.mlp.c_fc"] = ColwiseParallel()
        parallelize_plan[f"transformer.h.{i}.mlp.c_proj"] = RowwiseParallel()
        
        # Attention layers - can also be parallelized
        parallelize_plan[f"transformer.h.{i}.attn.c_attn"] = ColwiseParallel()
        parallelize_plan[f"transformer.h.{i}.attn.c_proj"] = RowwiseParallel()
    
    return parallelize_plan


def create_model_with_tensor_parallelism(model_name="gpt2", device_mesh=None, rank=0):
    """
    Create and parallelize a transformer model using tensor parallelism.
    """
    if rank == 0:
        print(f"\n{'='*70}")
        print("Loading Model with Tensor Parallelism")
        print(f"{'='*70}")
    
    # Load model configuration and adjust for demo
    config = AutoConfig.from_pretrained(model_name)
    config.n_layer = 6  # Use 6 layers for demo (instead of 12)
    config.n_head = 8
    config.n_embd = 512
    
    if rank == 0:
        print(f"Model: {model_name}")
        print(f"Layers: {config.n_layer}")
        print(f"Hidden size: {config.n_embd}")
        print(f"Attention heads: {config.n_head}")
    
    # Create model from config
    model = AutoModelForCausalLM.from_config(config)
    
    # Get parallelization plan
    parallelize_plan = get_parallelization_plan(model, num_layers=config.n_layer)
    
    if rank == 0:
        print(f"\nParallelizing {len(parallelize_plan)} layers...")
        print("Plan:")
        for layer_name, strategy in list(parallelize_plan.items())[:4]:
            print(f"  - {layer_name}: {strategy.__class__.__name__}")
        print(f"  ... and {len(parallelize_plan) - 4} more layers")
    
    # Apply tensor parallelism
    model = parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )
    
    # Move model to GPU
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = model.to(device)
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {num_params:,}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
    
    return model, device


def train_step(model, tokenizer, optimizer, device, rank=0):
    """
    Perform one complete training step: forward, backward, and optimizer update.
    """
    model.train()
    
    # Prepare sample batch
    texts = [
        "The future of artificial intelligence is",
        "Machine learning models can learn to",
        "Deep neural networks are capable of",
        "Tensor parallelism enables training of",
    ]
    
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = input_ids.clone()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print("Training Step")
        print(f"{'='*70}")
        print(f"Batch size: {input_ids.shape[0]}")
        print(f"Sequence length: {input_ids.shape[1]}")
    
    # Forward pass
    if rank == 0:
        print("\n[1/4] Forward pass...")
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    
    if rank == 0:
        print(f"      Loss: {loss.item():.6f}")
    
    # Backward pass
    if rank == 0:
        print("\n[2/4] Backward pass (computing gradients)...")
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    if rank == 0:
        print("\n[3/4] Gradient clipping...")
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    if rank == 0:
        print(f"      Gradient norm: {grad_norm:.6f}")
    
    # Optimizer step
    if rank == 0:
        print("\n[4/4] Optimizer step (updating weights)...")
    
    optimizer.step()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Step Complete! Final Loss: {loss.item():.6f}")
        print(f"{'='*70}\n")
    
    return loss.item()


def main():
    """Main function to demonstrate tensor parallelism."""
    try:
        # Setup distributed training
        rank, world_size, local_rank = setup_distributed()
        
        if rank == 0:
            print("\n" + "="*70)
            print("PyTorch Tensor Parallelism with parallelize_module")
            print("="*70)
            print(f"World size (number of GPUs): {world_size}")
            print(f"Global rank: {rank}")
            print(f"Local rank: {local_rank}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Initialize device mesh for tensor parallelism
        # This creates a 1D mesh across all GPUs
        if rank == 0:
            print("\nInitializing device mesh...")
        
        device_mesh = init_device_mesh("cuda", (world_size,))
        
        if rank == 0:
            print(f"Device mesh created: {device_mesh}")
        
        # Load tokenizer
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create model with tensor parallelism
        model, device = create_model_with_tensor_parallelism(
            model_name=model_name,
            device_mesh=device_mesh,
            rank=rank
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        if rank == 0:
            print("Optimizer created: AdamW with lr=5e-5\n")
        
        # Synchronize before training
        dist.barrier()
        
        # Perform training step
        loss = train_step(model, tokenizer, optimizer, device, rank)
        
        # Synchronize after training
        dist.barrier()
        
        if rank == 0:
            print("\n✓ Tensor parallelism example completed successfully!")
            print(f"  - Model was sharded across {world_size} GPUs")
            print(f"  - Each GPU held a portion of each weight matrix")
            print(f"  - Training step completed with loss: {loss:.6f}\n")
        
    except Exception as e:
        print(f"Error on rank {dist.get_rank() if dist.is_initialized() else 'unknown'}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    main()

