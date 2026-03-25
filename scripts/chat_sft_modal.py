"""
Modal training script for Chat SFT with LoRA.

This script runs SFT training directly on Modal instead of using the service client.

Prerequisites:
    1. Install modal: pip install modal
    2. Set up Modal secrets:
       - modal secret create huggingface-secret HF_TOKEN=<your-token>
       - modal secret create wandb-secret WANDB_API_KEY=<your-key>

Usage:
    # Run training on Modal
    modal run scripts/chat_sft_modal.py

    # Deploy as a persistent app
    modal deploy scripts/chat_sft_modal.py
"""
import os
import sys

import modal

# Modal app configuration
APP_NAME = "tinkerbell-chat-sft-local"
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
GPU_TYPE = "H100"
NUM_GPUS = 1
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 10
CHECKPOINT_INTERVAL = 100
HUGGINGFACE_REPO_ID = "shyamsn97/tinkerbell-chat-sft-local"

# Create Modal app
app = modal.App(name="tinkerbell-service")
env_variables = {
    "HF_TOKEN": os.environ.get("HF_TOKEN", None),
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "NCCL_DEBUG": "INFO",
    "TORCH_DISTRIBUTED_BACKEND": "nccl",
    "RAY_DEDUP_LOGS": "0",
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", None),
}
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python=f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    .apt_install("libnuma-dev", "build-essential", "clang")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch==2.4.0", extra_index_url="https://download.pytorch.org/whl/cu126"
    )
    .uv_pip_install(
        "pybase64",
        "zmq",
        "xformers",
        "transformers",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic>=2.0",
        "huggingface_hub",
        "hf_transfer",
        "peft",
        "datasets",
        "wandb",
        "tqdm",
    )
    .env(env_variables)
    .add_local_python_source("tinkerbell")
)

volume = modal.Volume.from_name("tinkerbell-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    volumes={"/checkpoints": volume},
    timeout=86400,
)
def train():
    """Run SFT training on Modal."""
    import torch
    import torch.distributed as dist
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import datasets
    from typing import cast
    import wandb
    import time
    
    from tinkerbell.renderer import Renderer, TrainOnWhat
    from tinkerbell.types.lora_config import LoraConfig
    from tinkerbell.types.datum import Datum
    from tinkerbell.training.llm import LLM
    from tinkerbell.training.loss import cross_entropy_loss

    print(f"Starting training with model: {MODEL_ID}")
    print(f"GPU: {GPU_TYPE}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    
    # Initialize distributed for single GPU (required by some internal code)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    # Create dataset wrapper
    class DatumDataset(Dataset):
        """PyTorch Dataset wrapper for list of Datum objects."""
        
        def __init__(self, datums: list[Datum]):
            self.datums = datums
        
        def __len__(self) -> int:
            return len(self.datums)
        
        def __getitem__(self, idx: int) -> Datum:
            return self.datums[idx]

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
    dataset = cast(datasets.DatasetDict, dataset)
    train_dataset = dataset["train"]

    # Build training samples
    print("Building training samples...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    renderer = Renderer(tokenizer)
    datums = renderer.build_chat_samples(
        messages=train_dataset["messages"],
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        mask_value=-100,
    )

    # Filter by sequence length
    original_count = len(datums)
    datums = [
        datum for datum in datums 
        if len(datum.model_input.input_ids) <= MAX_SEQUENCE_LENGTH
    ]
    filtered_count = len(datums)
    print(f"Filtered dataset: {original_count} -> {filtered_count} samples (keeping sequences <= {MAX_SEQUENCE_LENGTH} tokens)")
    if original_count > 0:
        print(f"Removed {original_count - filtered_count} samples ({100 * (original_count - filtered_count) / original_count:.1f}%)")

    # LoRA configuration
    lora_config = LoraConfig(
        rank=64,
        train_unembed=False,
        train_mlp=True,
        train_attn=True,
    )

    # Initialize model with LoRA
    print("Initializing model...")
    model = LLM(
        rank=0,
        world_size=1,
        base_model=MODEL_ID,
        model_kwargs={"torch_dtype": "bfloat16"},
        lora_config=lora_config,
        adapter_name="default",
        initialize_base_model=False,
        enable_gradient_checkpointing=True,
    )
    model.train()
    device = torch.cuda.current_device()

    # Initialize optimizer
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.2f}%)")

    # Initialize wandb
    wandb.init(
        project="tinkerbell-chat-sft-local",
        config={
            "base_model": MODEL_ID,
            "gpu_type": GPU_TYPE,
            "lora_rank": lora_config.rank,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "adam",
            "num_epochs": NUM_EPOCHS,
            "dataset": "HuggingFaceH4/no_robots",
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
        }
    )

    # Training loop
    global_step = 0
    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epoch")
    
    for epoch in epoch_bar:
        dataloader = DataLoader(
            DatumDataset(datums), 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            collate_fn=lambda x: x
        )

        accumulated_losses: list[float] = []
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(batch_bar):
            # Push checkpoint periodically
            if batch_idx % CHECKPOINT_INTERVAL == 0 and batch_idx > 0:
                print(f"Pushing checkpoint at step {global_step}...")
                model.model.push_to_hub(
                    repo_id=HUGGINGFACE_REPO_ID,
                    token=os.getenv("HF_TOKEN"),
                    private=False,
                )
                volume.commit()

            # Zero gradients at start of accumulation cycle
            if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0:
                step_start_time = time.time()
                optimizer.zero_grad()

            # Forward pass
            padded = model.pad(batch, device)
            forward_output = model.forward(
                model_inputs=padded["model_input"],
                with_grad=True,
            )
            logprobs = forward_output["logprobs"]

            # Compute loss
            per_batch_losses = cross_entropy_loss(
                logprobs=logprobs,
                **padded["loss_fn_inputs"]
            )
            loss_mean = per_batch_losses.mean()

            # Scale loss for gradient accumulation
            scaled_loss = loss_mean / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            # Track losses
            accumulated_losses.extend([loss.item() for loss in per_batch_losses])

            # Clean up intermediate tensors
            del forward_output, padded, per_batch_losses, logprobs

            # Optimizer step after accumulating gradients
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()

                # Calculate time per step
                step_time = time.time() - step_start_time

                if accumulated_losses:
                    avg_loss = sum(accumulated_losses) / len(accumulated_losses)
                    batch_bar.set_description(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}, Time: {step_time:.2f}s")

                    wandb.log({
                        "loss": avg_loss,
                        "epoch": epoch,
                        "step": global_step,
                        "time_per_step": step_time,
                    })
                    accumulated_losses = []

                global_step += 1

        # Handle remaining batches
        if accumulated_losses:
            optimizer.step()
            step_time = time.time() - step_start_time
            avg_loss = sum(accumulated_losses) / len(accumulated_losses)
            print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}, Time: {step_time:.2f}s (final batch)")
            wandb.log({
                "loss": avg_loss,
                "epoch": epoch,
                "step": global_step,
                "time_per_step": step_time,
            })
            global_step += 1

    # Final checkpoint
    print("Pushing final checkpoint...")
    model.model.push_to_hub(
        repo_id=HUGGINGFACE_REPO_ID,
        token=os.getenv("HF_TOKEN"),
        private=False,
    )
    volume.commit()

    # Cleanup
    dist.destroy_process_group()
    wandb.finish()
    print("Training complete!")


@app.local_entrypoint()
def main():
    """Local entrypoint to trigger the training job."""
    print("Launching training on Modal...")
    train.remote()
    print("Training job launched!")

