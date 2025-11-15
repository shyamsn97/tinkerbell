from tinkerbell.client import TrainingClient, ServiceClient
from tinkerbell.types import ModalDeployConfig

deploy_config = ModalDeployConfig(
    gpu="A100",
    num_gpus=4,
    timeout=86400,
    container_idle_timeout=600,
    max_inputs=200,
    max_wait_time=600.0,
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
)

training_client.wait_until_ready()

print("Training client ready")
# After training, save weights and get inference client
sampling_client = training_client.save_weights_and_get_sampling_client(
    checkpoint_path="/models/saved-qwen",
    tp_size=2,
    # engine_kwargs={"max_model_len": 4096},
    wait_until_ready=True
)
print("Sampling client ready")
# Generate text
outputs = sampling_client.generate(
    prompts=["Tell me a story about"],
    max_tokens=100,
    temperature=0.7,
)
print("Outputs: ", outputs)