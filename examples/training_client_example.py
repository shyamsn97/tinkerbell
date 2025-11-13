"""
Example demonstrating the TrainingClient for distributed training.

This script shows how to:
1. Create training actors
2. Wait for them to be ready
3. Tokenize inputs
4. Perform training steps with forward-backward passes
5. Update model weights with optimizer
"""

from tinkerbell.client import TrainingClient

# Configuration
WORLD_SIZE = 4
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"
MODEL_NAME = "Qwen/Qwen3-0.6B"

# Define parallelization plan for tensor parallelism
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


def main():
    # Initialize the training client
    with TrainingClient(
        base_url="https://jesterlabs--training-service.modal.run",
        timeout=600.0,
    ) as client:

        print("=" * 60)
        print("Step 1: Creating training actors")
        print("=" * 60)

        # Create training actors
        response = client.create_training_actors(
            model_name=MODEL_NAME,
            world_size=WORLD_SIZE,
            master_addr=MASTER_ADDR,
            master_port=MASTER_PORT,
            rank=0,
            parallelize_plan=parallelize_plan,
        )
        print(f"Response: {response.message}")

        print("\n" + "=" * 60)
        print("Step 2: Waiting for actors to be ready")
        print("=" * 60)

        # Wait for actors to be ready
        client.wait_until_ready(MODEL_NAME, verbose=True)

        print("\n" + "=" * 60)
        print("Step 3: Loading tokenizer and preparing data")
        print("=" * 60)

        # Load tokenizer
        tokenizer = client.load_tokenizer(MODEL_NAME)

        # Tokenize training examples
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
        ]
        tokenized = client.tokenize(texts, tokenizer=tokenizer)

        # Extract individual batches
        batch1 = {k: v.slice(0).model_dump() for k, v in tokenized.items()}
        batch2 = {k: v.slice(1).model_dump() for k, v in tokenized.items()}

        print("\n" + "=" * 60)
        print("Step 4: Training iteration 1")
        print("=" * 60)

        # Perform training step
        losses = client.train_step(
            model_name=MODEL_NAME,
            batch_inputs=[batch1, batch2],
            verbose=True,
        )
        print(f"Average loss: {sum(losses) / len(losses):.4f}")

        print("\n" + "=" * 60)
        print("Step 5: Training iteration 2 (should show lower loss)")
        print("=" * 60)

        # Perform another training step to verify learning
        losses = client.train_step(
            model_name=MODEL_NAME,
            batch_inputs=[batch1, batch2],
            verbose=True,
        )
        print(f"Average loss: {sum(losses) / len(losses):.4f}")

        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)


if __name__ == "__main__":
    main()

