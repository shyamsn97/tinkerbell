from tinkerbell.renderer import Renderer
from tinkerbell.client import ServiceClient
from tinkerbell.types import ModalDeployConfig, LoraConfig
from tinkerbell.renderer import Renderer, TrainOnWhat
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import datasets
from typing import cast
import wandb
import os
import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
GPU_TYPE = "H100"
NUM_GPUS = 1
# Filter by sequence length to save memory
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 64  # Reduced from 64 - start small and increase if stable
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 10
CHECKPOINT_INTERVAL = 100
HUGGINGFACE_REPO_ID = "shyamsn97/tinkerbell-chat-sft"

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE, 
    num_gpus=NUM_GPUS,
    timeout=86400,
    container_idle_timeout=600,
    max_inputs=200,
    max_wait_time=1200.0,
)

# Define parallelization plan
parallelize_plan = {
    # Attention projections (all layers)
    "model.layers.*.self_attn.q_proj": "column",
    "model.layers.*.self_attn.k_proj": "column",
    "model.layers.*.self_attn.v_proj": "column",
    "model.layers.*.self_attn.o_proj": "row",
    # MLP projections (all layers)
    "model.layers.*.mlp.gate_proj": "column",
    "model.layers.*.mlp.up_proj": "column",
    "model.layers.*.mlp.down_proj": "row",
}

# Create training client and train
server_url = "https://jesterlabs--training-service.modal.run"
service_client = ServiceClient(server_url=server_url, timeout=600.0)
print("Service client initialized")
server_url = service_client.deploy_or_connect(deploy_config)
print("Server URL: ", server_url)

lora_config_1 = LoraConfig(
    rank=64,
    train_unembed=False,
    train_mlp=True,
    train_attn=True,
)

# Enable gradient checkpointing to save memory
# This trades compute for memory by recomputing activations during backward
training_client = service_client.create_training_client(
    model_id=MODEL_ID,
    model_name="qwen-lora",  # Give it a name for multi-adapter support
    tp_size=NUM_GPUS,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=False,
    lora_config=lora_config_1.model_dump(),
    adapter_name="task1_attention",  # Name the first adapter
    model_kwargs={
        "torch_dtype": "bfloat16",
        "gradient_checkpointing": True,  # Enable gradient checkpointing to save activation memory
        # "use_cache": False,  # Disable KV cache during training to save memory
        # "flash_attention": True,
    }
)

# create dataset
class DatumDataset(Dataset):
    """PyTorch Dataset wrapper for list of Datum objects."""
    
    def __init__(self, datums):
        self.datums = datums
    
    def __len__(self):
        return len(self.datums)
    
    def __getitem__(self, idx):
        return self.datums[idx]


dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
dataset = cast(datasets.DatasetDict, dataset)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# renderer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
renderer = Renderer(tokenizer)
datums = renderer.build_chat_samples(
    messages=train_dataset["messages"],
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    mask_value=-100,
)

original_count = len(datums)
# Use len(datum.model_input) since ModelInput has __len__ that returns input_ids length
datums = [
    datum for datum in datums 
    if len(datum.model_input.input_ids) <= MAX_SEQUENCE_LENGTH
]
filtered_count = len(datums)
print(f"Filtered dataset: {original_count} -> {filtered_count} samples (keeping sequences <= {MAX_SEQUENCE_LENGTH} tokens)")
if original_count > 0:
    print(f"Removed {original_count - filtered_count} samples ({100 * (original_count - filtered_count) / original_count:.1f}%)")

# Initialize wandb
wandb.init(
    project="tinkerbell-chat-sft",
    config={
        "model_id": MODEL_ID,
        "gpu_type": GPU_TYPE,
        "lora_rank": lora_config_1.rank,
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

# # Enable gradient checkpointing via forward_kwargs
# # This significantly reduces memory usage at the cost of ~20% slower training
# forward_kwargs = {
#     "use_cache": False,  # Disable KV cache
# }

# training loop with gradient accumulation
global_step = 0
epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epoch")
for epoch in epoch_bar:
    dataloader = DataLoader(
        DatumDataset(datums), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda x: x
    )

    # Track accumulated loss for logging
    accumulated_losses = []

    batch_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(batch_bar):
        if batch_idx % CHECKPOINT_INTERVAL == 0:
            training_client.push_to_hub(
                repo_id=HUGGINGFACE_REPO_ID,
                token=os.getenv("HF_TOKEN"),
                private=False,
            ).result()
        # Only zero gradients at the start of accumulation cycle
        if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0:
            training_client.zero_grad().result()

        # Forward and backward pass
        forward_backward_response = training_client.forward_backward(
            data=batch,
        ).result()

        # Accumulate losses for logging
        if forward_backward_response.loss:
            accumulated_losses.extend(forward_backward_response.loss)

        # Only optimize after accumulating gradients
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            training_client.optim_step(
                optimizer_params={
                    "name": "adam",
                    "lr": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                }
            ).result()

            # Log accumulated losses
            if accumulated_losses:
                avg_loss = sum(accumulated_losses) / len(accumulated_losses)
                batch_bar.set_description(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f})")

                # Log to wandb
                log_dict = {
                    "loss": avg_loss,
                    "epoch": epoch,
                    "step": global_step,
                }

                # Log gradient sums if available (from last batch in accumulation)
                if forward_backward_response.sum_gradient:
                    for param_name, grad_sum in forward_backward_response.sum_gradient.items():
                        log_dict[f"gradients/{param_name}"] = grad_sum
                
                wandb.log(log_dict)
                accumulated_losses = []  # Reset for next accumulation cycle
            
            global_step += 1
    
    # Handle remaining batches that don't complete an accumulation cycle
    if accumulated_losses:
        training_client.optim_step(
            optimizer_params={
                "name": "adam",
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            }
        ).result()
        
        avg_loss = sum(accumulated_losses) / len(accumulated_losses)
        print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f} (final batch)")
        
        log_dict = {
            "loss": avg_loss,
            "epoch": epoch,
            "step": global_step,
        }
        wandb.log(log_dict)
        global_step += 1