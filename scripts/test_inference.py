import httpx
import torch
from transformers import AutoTokenizer
from tinkerbell.types.data import TensorData

WORLD_SIZE = 4
MODEL_NAME = "Qwen/Qwen3-0.6B"

def check_actor_status(client: httpx.Client):
    """Poll until inference actor is ready."""
    import time

    while True:
        response = client.post("/get_inference_actor_status", json={
            "model_name": MODEL_NAME,
        })
        status_data = response.json()
        print(f"Inference actor status: {status_data['status']}")

        if status_data["status"] == "ready":
            print("Inference actor is ready!")
            break

        time.sleep(2)

def create_inference_actor(client: httpx.Client, model_path: str, tp_size: int, engine_kwargs: dict = {}):
    """Create inference actor on the server."""
    response = client.post("/create_inference_actor", json={
        "model_path": model_path,
        "tp_size": tp_size,
        "engine_kwargs": engine_kwargs,
    })
    print("Create inference actor response:", response.json())
    return response.json()

def generate(client: httpx.Client, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
    """Generate text from a prompt."""
    response = client.post("/generate", json={
        "model_name": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    print("Generate response:", response.json())
    return response.json()

if __name__ == "__main__":
    client = httpx.Client(
        base_url="https://jesterlabs--training-service.modal.run",
        timeout=600.0
    )

    create_inference_actor(client, MODEL_NAME, 2)
    check_actor_status(client)
    text = generate(client, "Explain how the human brain works.", max_tokens=128)
    print("Generated text:", text)