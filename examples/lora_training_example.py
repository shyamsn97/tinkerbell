"""
Example of using LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

This example demonstrates:
1. Creating training actors with LoRA configuration
2. Training with LoRA adapters
3. Saving LoRA checkpoints
4. Loading checkpoints for inference

LoRA significantly reduces the number of trainable parameters compared to
full fine-tuning, making it more memory and compute efficient.
"""

import asyncio

from tinkerbell.client.training import TrainingClient
from tinkerbell.types import LoraConfig
from tinkerbell.types.data import TensorData
from tinkerbell.types.datum import Datum
from tinkerbell.types.model_input import ModelInput


async def main():
    # Configuration
    server_url = "http://localhost:8000"
    model_id = "meta-llama/Llama-3.2-1B"  # or any HuggingFace model
    
    # Initialize the training client
    client = TrainingClient(
        server_url=server_url,
        model_id=model_id,
        timeout=600.0,
    )
    
    print("=" * 80)
    print("LoRA Training Example")
    print("=" * 80)
    
    # ===============================================================
    # Step 1: Create LoRA Configuration
    # ===============================================================
    print("\n[Step 1] Creating LoRA configuration...")
    
    lora_config = LoraConfig(
        rank=8,  # LoRA rank - higher = more parameters, better quality
        seed=42,  # Optional: for reproducible initialization
        train_unembed=True,  # Apply LoRA to the output embedding layer
        train_mlp=True,  # Apply LoRA to MLP/FFN layers
        train_attn=True,  # Apply LoRA to attention layers (Q, K, V, O projections)
    )
    
    print(f"LoRA Config: rank={lora_config.rank}, "
          f"train_attn={lora_config.train_attn}, "
          f"train_mlp={lora_config.train_mlp}, "
          f"train_unembed={lora_config.train_unembed}")
    
    # ===============================================================
    # Step 2: Create Training Actors with LoRA
    # ===============================================================
    print("\n[Step 2] Creating training actors with LoRA enabled...")
    
    # Note: We pass lora_config as a dict via the HTTP API
    # You can also pass the LoraConfig object directly if using the Python client
    import httpx
    response = httpx.post(
        f"{server_url}/create_training_actors",
        json={
            "world_size": 1,  # Number of GPUs
            "model_id": model_id,
            "model_kwargs": {
                "torch_dtype": "bfloat16",
            },
            "parallelize_plan": {},  # Can add tensor parallelism if needed
            "scheduler_params": {},
            "lora_config": lora_config.model_dump(),  # Pass LoRA config here
            "initialize_random_weights": False,
        },
        timeout=600.0,
    )
    response.raise_for_status()
    print(f"Response: {response.json()}")
    
    # Wait for actors to be ready
    print("\n[Step 3] Waiting for actors to initialize...")
    client.wait_until_ready(verbose=True)
    print("✓ Actors ready!")
    
    # ===============================================================
    # Step 4: Prepare Training Data
    # ===============================================================
    print("\n[Step 4] Preparing training data...")
    
    # Example: Simple instruction following
    prompts = [
        "Question: What is the capital of France? Answer:",
        "Question: What is 2+2? Answer:",
    ]
    
    completions = [
        " Paris",
        " 4",
    ]
    
    # Tokenize the data
    data_batch = []
    for prompt, completion in zip(prompts, completions):
        full_text = prompt + completion
        
        # Tokenize
        tokenized = client.tokenizer([full_text])
        input_ids = tokenized["input_ids"][0]
        
        # Create labels (mask the prompt tokens)
        prompt_tokens = client.tokenizer([prompt])["input_ids"][0]
        prompt_length = len(prompt_tokens)
        
        # Labels: -100 for prompt tokens (ignored), actual tokens for completion
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        
        # Create Datum object
        datum = Datum(
            model_input=ModelInput(
                tokens=input_ids,
                attention_mask=None,  # Optional: can add attention mask
                additional_inputs=None,
            ),
            loss_fn_inputs={
                "labels": TensorData.from_list(labels),
            },
        )
        data_batch.append(datum)
    
    print(f"Prepared {len(data_batch)} training examples")
    
    # ===============================================================
    # Step 5: Training Loop
    # ===============================================================
    print("\n[Step 5] Starting training...")
    
    num_steps = 5
    for step in range(num_steps):
        print(f"\n--- Training Step {step + 1}/{num_steps} ---")
        
        # Zero gradients
        client.zero_grad()
        
        # Forward + backward pass
        result = client.forward_backward(
            data=data_batch,
            forward_kwargs={},
            return_logprobs=False,
        )
        
        # Get the result
        loss_info = client.get_result(result.request_id)
        losses = loss_info.get("loss", [])
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        print(f"  Loss: {avg_loss:.4f} (per sample: {losses})")
        
        # Optimizer step
        client.optim_step(
            optimizer_params={
                "name": "adamw",
                "lr": 1e-4,  # Learning rate
                "weight_decay": 0.01,
            }
        )
        
        print(f"  ✓ Step {step + 1} complete")
    
    # ===============================================================
    # Step 6: Save LoRA Checkpoint
    # ===============================================================
    print("\n[Step 6] Saving LoRA checkpoint...")
    
    checkpoint_path = "/tmp/lora_checkpoint"
    save_response = client.save_checkpoint(checkpoint_path=checkpoint_path)
    print(f"  Checkpoint saved to: {checkpoint_path}")
    print(f"  Response: {save_response}")
    
    print("\n" + "=" * 80)
    print("LoRA Training Complete!")
    print("=" * 80)
    print(f"\nCheckpoint saved at: {checkpoint_path}")
    print("This checkpoint contains only the LoRA adapter weights,")
    print("not the full model. To use it:")
    print("  1. Load the base model")
    print("  2. Apply the LoRA adapters from the checkpoint")
    print("\nExample LoRA adapter files in checkpoint:")
    print("  - adapter_config.json")
    print("  - adapter_model.bin")
    print("  - tinkerbell_lora_info.json (metadata)")
    
    # Clean up
    client.close()


if __name__ == "__main__":
    asyncio.run(main())

