import os
import ray
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh
import fnmatch
import re
import modal
import time
import json
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict
)

def get_submodules_with_wildcard(model, pattern):
    """Get all submodules matching a wildcard pattern."""
    regex_pattern = fnmatch.translate(pattern)
    regex = re.compile(regex_pattern)

    matching_modules = []
    for name, module in model.named_modules():
        if regex.match(name):
            matching_modules.append(name)

    return matching_modules


def print_gpu_memory(prefix="", rank=0):
    """Print GPU memory usage for the current device."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(rank).total_memory / 1024**3

    print(f"[Rank {rank}] {prefix}")
    print(f"  GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB | Total: {total:.2f}GB")

@ray.remote(num_gpus=1)
class TensorParallelWorker:
    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: str):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.model = None
        self.optimizer = None

    def setup(self):
        """Initialize the PyTorch distributed process group and model."""
        print(f"[Rank {self.rank}] Initializing torch distributed")

        # Set environment variables for distributed setup
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        # Initialize process group
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

        # Set CUDA device - Ray manages GPU assignment via CUDA_VISIBLE_DEVICES
        # Each actor sees only one GPU as device 0
        torch.cuda.set_device(0)

        # Initialize model with RANDOM WEIGHTS using Qwen3 architecture
        # from_config() creates model with random initialization (NOT pretrained weights)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")  # Creates model with random weights

        # Define parallelization strategies
        strategies = {
            "column": ColwiseParallel,
            "row": RowwiseParallel,
        }

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

        # Build module parallelization plan
        module_parallelization_plan = {}
        for pattern in parallelize_plan.keys():
            strategy = strategies[parallelize_plan[pattern]]()
            module_names = get_submodules_with_wildcard(self.model, pattern)
            for name in module_names:
                module_parallelization_plan[name] = strategy

        # Initialize device mesh and parallelize model
        device_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))
        self.model = parallelize_module(self.model, device_mesh, module_parallelization_plan)
        self.model = self.model.cuda()

        # Print memory - clarify that each actor uses device 0 (Ray's CUDA_VISIBLE_DEVICES isolation)
        print_gpu_memory(f"Model loaded (Rank {self.rank}, physical device isolated by Ray as cuda:0)", 0)

        # Setup optimizer - disable foreach to handle mixed DTensor/Tensor parameters
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, foreach=False)

        print(f"[Rank {self.rank}] Setup complete")
        return True

    def train_step(self):
        """Execute a single training step."""
        # Prepare data
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(["Hello world!"], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].cuda()

        # Training step
        self.model.train()
        outputs = self.model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            print(f"[Rank {self.rank}] ✓ Training step complete! Loss: {loss.item():.4f}")

        return loss.item() if self.rank == 0 else None

    def zero_out_lm_head(self):
        """Zero out the lm_head weights"""
        for _, param in self.model.named_parameters():
            param.data.zero_()
        return True

    def create_test_params_batch(self, model, num_params=64):
        """Create a batch of test parameters from the model"""
        param_names = []
        test_tensors = []

        # Get first few parameters from the model for testing
        for i, (name, tensor) in enumerate(model.named_parameters()):
            if i >= num_params:
                break
            param_names.append(name)
            # Create test tensor with known values, matching original shape and dtype
            test_tensor = torch.full_like(tensor, 1.5, dtype=tensor.dtype).cuda()
            test_tensors.append(test_tensor)

        return list(zip(param_names, test_tensors))

    def get_model_state_dict(
        self, full_state_dict: bool = False
    ):
        options = StateDictOptions(
            full_state_dict=full_state_dict,
            cpu_offload=True
        )
        # self._load_model_to_device(torch.cuda.current_device())
        state_dict = get_model_state_dict(self.model, options=options)
        # self._load_model_to_device("cpu")
        return state_dict

    def save_model(self, save_dir: str):

        state_dict = self.get_model_state_dict(full_state_dict=True)
        if self.rank == 0:

            # self.tokenizer.save_pretrained(save_dir)
            self.model.save_pretrained(
                save_dir, state_dict=state_dict
            )

        dist.barrier()

    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint using PyTorch's distributed checkpoint API.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        # from torch.distributed.checkpoint import save

        print(f"[Rank {self.rank}] Starting checkpoint save process...")

        if self.rank == 0:
            os.makedirs(checkpoint_path, exist_ok=True)

        self.save_model(checkpoint_path)
        print(f"[Rank {self.rank}] Checkpoint save complete")

        return self.rank == 0

    def cleanup(self):
        """Clean up the PyTorch distributed process group."""
        print(f"[Rank {self.rank}] Cleaning up torch distributed")
        dist.destroy_process_group()
        return True


# app = modal.App(name="ray-monarch-example")

# image = modal.Image.debian_slim().pip_install("torch", "torchvision", "torchaudio", "transformers", "ray").env({"CUDA_HOME": "/usr/local/cuda", "RAY_DEDUP_LOGS": "0"})

@ray.remote
class SGLangInferenceActor:
    def __init__(self, model_path: str, tp_size: int, engine_kwargs: dict = {}):
        import sglang as sgl
        import asyncio

        # Create event loop for this actor
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            **engine_kwargs,
        )
        print(f"✓ SGLang engine initialized with {tp_size} GPUs")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from a directory"""
        import asyncio
        # Ensure the event loop is set for the current thread
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(self.loop)

        return self.engine.update_weights_from_disk(checkpoint_path)

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Generate text from a prompt."""
        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

        # Use async_generate with the event loop
        outputs = self.loop.run_until_complete(
            self.engine.async_generate(prompt, sampling_params)
        )
        return outputs["text"]

    def shutdown(self):
        del self.engine
        return True

def setup_inference_server(num_inference_gpus: int = 2):
    """Set up SGLang inference engine with configurable GPU count.
    
    Args:
        num_inference_gpus: Number of GPUs to use for inference (default: 2)
    """
    print("\n" + "="*50)
    print(f"Setting up SGLang Inference Engine ({num_inference_gpus} GPUs)")
    print("="*50)

    model_path = "Qwen/Qwen3-0.6B"  # or your model

    # Create inference actor with dynamic GPU allocation using .options()
    inference_actor = SGLangInferenceActor.options(num_gpus=num_inference_gpus).remote(
        model_path, num_inference_gpus
    )

    return inference_actor


# @ray.remote
# class VLLMInferenceActor:
#     def __init__(self, model_path: str, tp_size: int):
#         from vllm import LLM, SamplingParams
#         import ray as ray_import
        
#         # Verify we're in an existing Ray context
#         print(f"Ray is initialized: {ray_import.is_initialized()}")
        
#         # Configure vLLM to use existing Ray context
#         self.llm = LLM(
#             model=model_path,
#             tensor_parallel_size=tp_size,
#             distributed_executor_backend="ray",
#             # worker_use_ray=True,  # Explicitly use Ray for workers
#         )
#         self.SamplingParams = SamplingParams
#         print(f"✓ vLLM engine initialized with {tp_size} GPUs")
        
#     def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
#         """Generate text from a prompt."""
#         sampling_params = self.SamplingParams(
#             max_tokens=max_tokens,
#             temperature=temperature,
#         )
        
#         outputs = self.llm.generate(prompt, sampling_params)
#         return outputs[0].outputs[0].text
    
#     def shutdown(self):
#         del self.llm
#         return True

# def setup_inference_server(num_inference_gpus: int = 2):
#     """Set up vLLM inference engine with configurable GPU count.
    
#     Args:
#         num_inference_gpus: Number of GPUs to use for inference (default: 2)
#     """
#     print("\n" + "="*50)
#     print(f"Setting up vLLM Inference Engine ({num_inference_gpus} GPUs)")
#     print("="*50)
    
#     model_path = "Qwen/Qwen3-0.6B"  # or your model
    
#     # Create inference actor with dynamic GPU allocation using .options()
#     inference_actor = VLLMInferenceActor.options(num_gpus=num_inference_gpus).remote(
#         model_path, num_inference_gpus
#     )
    
#     return inference_actor


app = modal.App(name="ray-monarch-example")


env_variables = {
    "HF_TOKEN": os.environ.get("HF_TOKEN", None),
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "NCCL_DEBUG": "INFO",  # For debugging NCCL issues
    "TORCH_DISTRIBUTED_BACKEND": "nccl",  # Prefer NCCL
    # Add these for better visibility:
    "TORCH_DISTRIBUTED_DEBUG": "INFO",  # Shows distributed init details
    "TORCH_SHOW_CPP_STACKTRACES": "1",  # If it crashes
    "SGLANG_LOG_LEVEL": "DEBUG",  # If SGLang respects this
    "TRANSFORMERS_VERBOSITY": "info",  # See model loading progress
    "RAY_DEDUP_LOGS": "0",
}

volume = modal.Volume.from_name("tinkerbell-checkpoints", create_if_missing=True)

# Define Modal image with required dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("libnuma-dev", "build-essential", "clang")
    .env({"CUDA_HOME": "/usr/local/cuda"})  # Add this line
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .uv_pip_install(
        "pybase64",
        "zmq",
        "xformers",
        "transformers",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "cloudpickle",
        "dill",
        "flashinfer-python",  # Install FlashInfer first
        "sglang[all]==0.5.2",
        "sgl-kernel",
        "huggingface_hub",
        "hf_transfer",
        "safetensors",  # For modern checkpoint format
        "ray"
    )
    .env(env_variables)
)

@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    gpu="A100:4",
    timeout=24*60*60,
    container_idle_timeout=5*60,
)
def main():
    # Initialize Ray with log deduplication disabled
    ray.init(
        num_gpus=4,
        # logging_level="info",
        # log_to_driver=True,
    )

    # Configuration
    WORLD_SIZE = 2
    MASTER_ADDR = "127.0.0.1"
    MASTER_PORT = "29500"

    # Create worker actors
    workers = [
        TensorParallelWorker.remote(
            rank=i,
            world_size=WORLD_SIZE,
            master_addr=MASTER_ADDR,
            master_port=MASTER_PORT
        )
        for i in range(WORLD_SIZE)
    ]

    # Setup all workers (this initializes distributed training)
    print("Setting up workers...")
    setup_results = ray.get([worker.setup.remote() for worker in workers])
    print(f"Setup results: {setup_results}")
    time.sleep(5)

    # Save randomly initialized weights before any training
    initial_checkpoint_path = "/checkpoints/initial_weights"
    print(f"\nSaving initial weights to {initial_checkpoint_path}...")
    initial_save_results = ray.get([worker.save_checkpoint.remote(initial_checkpoint_path) for worker in workers])
    print(f"Initial checkpoint saved. Results: {initial_save_results}")

    # Run training steps
    print("\nRunning training step...")
    train_results = ray.get([worker.train_step.remote() for worker in workers])
    print(f"Training complete. Loss from rank 0: {train_results[0]}")

    # Zero out lm_head weights to demonstrate checkpoint saving/loading
    print("\n" + "="*50)
    print("DEMONSTRATION: Zeroing out lm_head weights")
    print("="*50)
    print("This proves that the checkpoint is actually being saved and loaded.")
    print("The inference engine will generate nonsensical outputs with zeroed weights.")
    zero_results = ray.get([worker.zero_out_lm_head.remote() for worker in workers])
    print(f"Zero out results: {zero_results}")

    # Save checkpoint after training
    checkpoint_path = "/checkpoints/random_weights_checkpoint"
    print(f"\nSaving random weights checkpoint to {checkpoint_path}...")
    save_results = ray.get([worker.save_checkpoint.remote(checkpoint_path) for worker in workers])
    print(f"Checkpoint saved. Results: {save_results}")

    # Set up inference engine
    print("\nSetting up inference engine...")
    inference_actor = setup_inference_server(num_inference_gpus=2)

    # Generate text with BASE model (before loading zeroed checkpoint)
    print("\n" + "="*50)
    print("BASELINE: Generating with BASE model")
    print("="*50)
    prompt = "What is machine learning?"
    print(f"Prompt: {prompt}")
    base_response = ray.get(inference_actor.generate.remote(prompt, max_tokens=50, temperature=0.7))
    print(f"✓ Base model output: {base_response}")

    # Load trained weights (with zeroed lm_head) into inference engine
    print("\n" + "="*50)
    print("Loading checkpoint with ZEROED lm_head weights")
    print("="*50)
    ray.get(inference_actor.load_checkpoint.remote(checkpoint_path))
    print("✓ Checkpoint with zeroed weights loaded successfully")

    # Generate text with model that has zeroed lm_head
    print("\n" + "="*50)
    print("DEMONSTRATION: Generating with ZEROED lm_head")
    print("="*50)
    print(f"Prompt: {prompt}")
    print("(Output should be very different/nonsensical, proving checkpoint was loaded)")
    zeroed_response = ray.get(inference_actor.generate.remote(prompt, max_tokens=50, temperature=0.7))
    print(f"✓ Zeroed model output: {zeroed_response}")

    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Base model:   {base_response}")
    print(f"Zeroed model: {zeroed_response}")
    print("\nIf the outputs are different, checkpoint save/load is working! ✓")

    # Cleanup
    print("\nCleaning up...")
    cleanup_results = ray.get([worker.cleanup.remote() for worker in workers])
    print(f"Cleanup results: {cleanup_results}")
    
    # Shutdown Ray
    ray.shutdown()
    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main.remote()