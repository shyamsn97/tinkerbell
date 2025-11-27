"""
Example demonstrating async training API usage.

This example shows how to use the async methods for training,
which can be more efficient when performing multiple operations concurrently.
"""

import asyncio
from tinkerbell.client import ServiceClient
from tinkerbell.renderer import Renderer, TrainOnWhat
from tinkerbell.types import LoraConfig


async def async_training_example():
    """Example of using async training methods."""
    
    # Initialize service client
    service = ServiceClient(server_url="http://localhost:8000")
    
    # Create training client with LoRA
    lora_config = LoraConfig(
        rank=8,
        seed=42,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    
    training_client = service.create_training_client(
        model_id="meta-llama/Llama-3.2-1B",
        tp_size=1,
        lora_config=lora_config.model_dump(),
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    training_client.wait_until_ready()
    
    # Prepare training data
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
    
    # Use async context manager for automatic cleanup
    async with training_client:
        # Training loop using async methods
        for step in range(10):
            # Zero gradients
            # First await submits the request, second await waits for completion
            zero_grad_future = await training_client.zero_grad_async()
            await zero_grad_future
            
            # Forward and backward pass
            # First await submits request and ensures ordering
            fb_future = await training_client.forward_backward_async(
                data=training_data,
                forward_kwargs={},
            )
            # Second await waits for computation to finish and guarantees gradients are accumulated
            result = await fb_future
            
            losses = result.loss
            if losses:
                avg_loss = sum(losses) / len(losses)
                print(f"Step {step}, Loss: {avg_loss:.4f}")
            
            # Optimizer step
            optim_future = await training_client.optim_step_async(
                optimizer_params={
                    "name": "adamw",
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                }
            )
            await optim_future
        
        # Save checkpoint
        checkpoint_future = await training_client.save_checkpoint_async("/tmp/async_model")
        checkpoint_response = await checkpoint_future
        print(f"Checkpoint saved: {checkpoint_response.checkpoint_path}")


async def parallel_operations_example():
    """Example showing how to perform multiple async operations in parallel."""
    
    service = ServiceClient(server_url="http://localhost:8000")
    training_client = service.create_training_client(
        model_id="meta-llama/Llama-3.2-1B",
        tp_size=1,
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    training_client.wait_until_ready()
    
    # Prepare multiple batches
    tokenizer = training_client.get_tokenizer()
    renderer = Renderer(tokenizer)
    
    batch1_conversations = [
        [
            {"role": "user", "content": "What is 1 + 1?"},
            {"role": "assistant", "content": "The answer is 2."},
        ],
    ]
    
    batch2_conversations = [
        [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ],
    ]
    
    batch1 = renderer.build_chat_examples(
        conversations=batch1_conversations,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value=-100,
    )
    
    batch2 = renderer.build_chat_examples(
        conversations=batch2_conversations,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        mask_value=-100,
    )
    
    async with training_client:
        # Zero gradients
        zero_grad_future = await training_client.zero_grad_async()
        await zero_grad_future
        
        # Submit multiple forward_backward requests in parallel
        # First await ensures requests are submitted and ordered correctly
        fb_future1 = await training_client.forward_backward_async(data=batch1)
        fb_future2 = await training_client.forward_backward_async(data=batch2)
        
        # Wait for both to complete
        result1 = await fb_future1
        result2 = await fb_future2
        
        print(f"Batch 1 loss: {sum(result1.loss) / len(result1.loss):.4f}")
        print(f"Batch 2 loss: {sum(result2.loss) / len(result2.loss):.4f}")
        
        # Optimizer step
        optim_future = await training_client.optim_step_async(
            optimizer_params={"name": "adamw", "lr": 1e-4}
        )
        await optim_future


if __name__ == "__main__":
    # Run the async training example
    asyncio.run(async_training_example())
    
    # Or run the parallel operations example
    # asyncio.run(parallel_operations_example())

