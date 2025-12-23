from typing import cast
# from datasets import DatasetDict, load_dataset
import gymnasium as gym
from tinkerbell.types import ModelInput, TensorData
from pydantic import BaseModel
from tinkerbell.types import ModalDeployConfig
from tinkerbell.client import ServiceClient
from tinkerbell.renderer import Renderer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B"
GPU_TYPE = "A100"
NUM_GPUS = 1

deploy_config = ModalDeployConfig(
    gpu=GPU_TYPE, 
    num_gpus=NUM_GPUS,
    timeout=86400,
    container_idle_timeout=600,
    max_inputs=200,
    max_wait_time=1200.0,
)

service_client = ServiceClient.deploy_or_connect(deploy_config)

sampling_client = service_client.create_sampling_client(
    base_model=BASE_MODEL,
    tp_size=NUM_GPUS,
    model_name="qwen3-06b",
)

sampling_client.wait_until_ready()

# Full model: General knowledge (uses all parameters)
full_model_conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about machine learning."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about bayesian inference."},
    ],
]

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
renderer = Renderer(tokenizer)

input_data = renderer.build_chat_samples(
    messages=full_model_conversations,
    mask_value=-100,
)

sampling_params = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    # "top_k": 50,
    # "return_text_in_logprobs": False
}
sample_futures = sampling_client.sample(
    input_ids=input_data[0].model_input.input_ids,
    sampling_params=sampling_params,
    return_logprob=True,
    top_logprobs_num=256,
)

out = sample_futures.result()

print("================================================")
print("Output:")
print(out.outputs)
print("================================================")
print("Logprobs:")
print(out.logprobs)
print("================================================")
print("Top Logprobs:")
print(out.top_logprobs[0])
print("================================================")
print("Output Token IDs:")
print(out.output_token_ids)
print("================================================")
