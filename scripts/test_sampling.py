from tinkerbell.types import ModalDeployConfig
from tinkerbell.client import ServiceClient
from tinkerbell.utils import save_dict_to_json

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
        {"role": "system", "content": "You are a helpful assistant. Put your thoughts in <think></think> tags. Respond with <answer></answer> tags."},
        {"role": "user", "content": "Tell me about machine learning."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant. Put your thoughts in <think></think> tags. Respond with <answer></answer> tags."},
        {"role": "user", "content": "Tell me about bayesian inference."},
    ],
]

# Convert chat messages to formatted text using the tokenizer's chat template
tokenizer = sampling_client.get_tokenizer()
formatted_prompts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    for messages in full_model_conversations
]
tokenized_conversations = [tokenizer.encode(prompt) for prompt in formatted_prompts]
# tokenized_conversations = sampling_client.render(
#     full_model_conversations,
#     mode="inference",
#     add_generation_prompt=True,
#     continue_final_message=False,
# )

# print("Input Data:")
# print(sampling_client.get_tokenizer().decode(tokenized_conversations[0].get_input_ids().tolist()))

sampling_params = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    # "top_k": 50,
    # "return_text_in_logprobs": False
}
sample_futures = sampling_client.sample_batch(
    batch_kwargs=[
        {
            "sampling_params": sampling_params,
            "input_ids": tokenized_conversation,
        }
    for tokenized_conversation in tokenized_conversations],
)

# sample_batch returns a list of futures - get results for each
results = [future.result() for future in sample_futures]

save_dict_to_json(results[0].model_dump(), "results.json")

for i, out in enumerate(results):
    print("================================================")
    print("================================================")
    print(f"Output {i + 1}:")
    print(out.output)
    print("================================================")
    print("Top Logprobs:")
    print(out.logprobs.logprobs.shape)
    print("================================================")
    print("Output Token IDs:")
    print(out.output_token_ids)
    print("================================================")

    # Verify output_token_ids decode to the same text
    decoded = tokenizer.decode(out.output_token_ids, skip_special_tokens=False)
    match = decoded == out.output
    print(f"Token decode: {'✓ MATCH' if match else '✗ MISMATCH'}")
    if not match:
        print(f"  Original: {repr(out.output)}")
        print(f"  Decoded:  {repr(decoded)}")
    assert match, "Output token IDs don't decode to original text!"
    print("Verified! ✓")
