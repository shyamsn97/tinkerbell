from tinkerbell.client import TrainingClient, ServiceClient
from tinkerbell.types import ModalDeployConfig

import time
import concurrent.futures
from functools import partial

from tqdm import tqdm

deploy_config = ModalDeployConfig(
    gpu="A100",
    num_gpus=4,
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
service_client = ServiceClient(timeout=600.0)
server_url = service_client.deploy(deploy_config)
print("Deployed to: ", server_url)

training_client = service_client.create_training_client(
    model_id="Qwen/Qwen3-0.6B",
    tp_size=2,
    parallelize_plan=parallelize_plan,
    initialize_random_weights=True,
)
training_client.wait_until_ready()

print("Training client ready")
# After training, save weights and get inference client
sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/saved-qwen",
    tp_size=2,
    wait_until_ready=True
)
print("Sampling client ready")
# Generate text

messages = [
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the largest planet in our solar system?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "How many continents are there?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the chemical symbol for gold?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "In what year did World War II end?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the speed of light?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "Who painted the Mona Lisa?"},
        {"role": "system", "content": "<think>"},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Wrap thinking processes in <think> and </think> tags. Answer should be wrapped in <answer> and </answer> tags."},
        {"role": "user", "content": "What is the smallest unit of matter?"},
        {"role": "system", "content": "<think>"},
    ],
]

formatted_messages = training_client.tokenizer.apply_chat_template(messages)

print("=" * 70)
print("Multithreaded request to generate...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(sampling_client.generate, prompts=[formatted_messages[i]], sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multithreaded requests"):
        outputs = future.result()
print(f"Time taken: {time.time() - start_time} seconds")

print("Multithreaded request to generate run # 2...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(sampling_client.generate, prompts=[formatted_messages[i]], sampling_params={"max_new_tokens": 100, "temperature": 0.7}) for i in range(len(messages))]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Multithreaded requests"):
        outputs = future.result()
print("Outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)

print("=" * 70)
print("Sequential request to generate...")
start_time = time.time()
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    outputs = sampling_client.generate(
        prompts=[formatted_messages[i]],
        sampling_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )

print("Sequential request to generate run # 2...")
for i in tqdm(range(len(messages)), desc="Sequential requests"):
    outputs = sampling_client.generate(
        prompts=[formatted_messages[i]],
        sampling_params={
            "max_new_tokens": 100,
            "temperature": 0.7,
        },
    )
print("Outputs: ", outputs)
print(f"Time taken: {time.time() - start_time} seconds")
print("=" * 70)
