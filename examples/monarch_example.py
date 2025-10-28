from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import modal
from modal import runner
import json
import re
import fnmatch
import torch

def get_submodules_with_wildcard(model, pattern):
    """
    Get all submodules matching a wildcard pattern.

    Args:
        model: PyTorch model
        pattern: Pattern with wildcards (e.g., "model.layers.*.self_attn.q_proj")

    Returns:
        List of (name, module) tuples matching the pattern
    """
    # Convert wildcard pattern to regex
    regex_pattern = fnmatch.translate(pattern)
    regex = re.compile(regex_pattern)

    matching_modules = []
    for name, module in model.named_modules():
        if regex.match(name):
            matching_modules.append(name)

    return matching_modules


def print_gpu_memory(prefix="", rank=0, print_method=print):
    """Print GPU memory usage for the current device."""
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(rank).total_memory / 1024**3

    print_method(f"[Rank {rank}] {prefix}")
    print_method(f"  GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_allocated:.2f}GB | Total: {total:.2f}GB")



image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"  # Changed from 12.1.0 to 12.4.0
    ).apt_install("rdma-core","libibverbs1","libnuma-dev","git","build-essential", "clang")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    })
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",  # Changed from cu121 to cu124
    )
    .uv_pip_install(
        "cloudpickle>=3.0.0",
        "transformers",
        "fastapi",
        "uvicorn",
        "pydantic>=2.0.0",
        "dill",
        "pybase64",
        "partial_json_parser",
        "psutil",
        "sentencepiece",
        "openai",
        "huggingface_hub",
        "hf_transfer",
        "torchmonarch-nightly",
        "flashinfer-python",
        "sglang[all]",
        "sgl-kernel",
        "uvloop",
    )
)
    # .pip_install(
    #     "git+ssh://git@github.com/meta-pytorch/monarch.git"
    # )
    # .run_commands(
    #     "pip install ninja",
    #     "pip install --no-build-isolation --no-cache-dir sgl-kernel",
    # )


app = modal.App("monarch-example")

@app.function(image=image, gpu="H100:6")
def main():
    import math
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from monarch.actor import Actor, current_rank, current_size, endpoint, this_host
    import asyncio
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel, SequenceParallel
    from torch.distributed.device_mesh import init_device_mesh

    NUM_ACTORS = 4
    NUM_INFERENCE_ACTORS = 2
    WORLD_SIZE = NUM_ACTORS

    class TensorParallelActor(Actor):
        def __init__(self):
            self.rank = current_rank().rank
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
            config.n_layer = 4  # Small model for demo
            self.model = AutoModelForCausalLM.from_config(config)

        def _rprint(self, msg):
            """Helper method to print with rank information."""
            print(f"{self.rank=} {msg}")


        @endpoint
        async def setup(self):
            """Initialize the PyTorch distributed process group."""
            self._rprint("Initializing torch distributed")
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # initialize the process group
            rank = self.rank
            dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
            self._rprint("Finished initializing torch distributed")

            torch.cuda.set_device(rank)

            strategies = {
                "column": ColwiseParallel,
                "row": RowwiseParallel,
                "sequence": SequenceParallel,
            }

            # Define parallelization plan
            parallelize_plan = {
                # Note: Embedding layer is NOT parallelized - it remains replicated
                
                # Attention projections (all layers)
                "model.layers.*.self_attn.q_proj": "column",
                "model.layers.*.self_attn.k_proj": "column",
                "model.layers.*.self_attn.v_proj": "column",
                "model.layers.*.self_attn.o_proj": "row",

                # MLP projections (all layers)
                "model.layers.*.mlp.gate_proj": "column",
                "model.layers.*.mlp.up_proj": "column",
                "model.layers.*.mlp.down_proj": "row",
                
                # Note: lm_head is NOT parallelized to avoid issues with loss computation
                # "lm_head": "row",
            }

            module_parallelization_plan = {}

            for pattern in parallelize_plan.keys():
                strategy = strategies[parallelize_plan[pattern]]()
                module_names = get_submodules_with_wildcard(self.model, pattern)
                for name in module_names:
                    module_parallelization_plan[name] = strategy

            device_mesh = init_device_mesh("cuda", (1, WORLD_SIZE), mesh_dim_names=("dp", "tp"))
            self.model = parallelize_module(self.model, device_mesh["tp"], module_parallelization_plan)
            self.model = self.model.cuda()

            # Print
            print_gpu_memory(f"Model loaded on GPU {rank}", rank, self._rprint)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, foreach=False)

        @endpoint
        async def step(self):
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
                self._rprint(f"✓ Training step complete! Loss: {loss.item():.4f}")

        @endpoint
        async def cleanup(self):
            """Clean up the PyTorch distributed process group."""
            self._rprint("Cleaning up torch distributed")
            dist.destroy_process_group()

    async def create_tp_actors():
        """Create the process mesh and spawn TP actors."""
        # Spawn a process mesh
        local_proc_mesh = this_host().spawn_procs(per_host={"gpus": WORLD_SIZE})
        # Spawn our actor mesh on top of the process mesh
        tp_actor = local_proc_mesh.spawn("tp_actor", TensorParallelActor)
        return tp_actor, local_proc_mesh


    class SGLangActor:
        def __init__(self, model_path: str = "Qwen/Qwen3-0.6B"):
            from sglang import Engine
            self.engine = Engine(
                model_path=model_path,
                tp_size=NUM_INFERENCE_ACTORS,
                log_level="info",
            )

        def generate(self, prompts: List[str], max_tokens: int = 128, temperature: float = 0.7):
            """Generate text using the SGLang engine."""
            if self.engine is None:
                raise RuntimeError("Engine not initialized. Call setup() first.")
            
            # Only rank 0 should handle the actual generation
            print(f"Generating for {len(prompts)} prompts")
            
            outputs = self.engine.generate(
                prompts=prompts,
                sampling_params={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            
            results = [output["text"] for output in outputs]
            print(f"✓ Generation complete for {len(results)} prompts")
            return results

    # async def run_sglang_example():
    #     """Run the SGLang inference example."""
    #     sglang_actor, sglang_mesh = await create_sglang_actors()
        
    #     # Setup the engines
    #     await sglang_actor.setup.call(
    #         model_path="Qwen/Qwen3-0.6B",
    #         tp_size=NUM_INFERENCE_ACTORS
    #     )
        
    #     # Generate some text (only rank 0 will return results)
    #     prompts = [
    #         "Hello, my name is",
    #         "The meaning of life is",
    #         "In a world where AI"
    #     ]
        
    #     results = await sglang_actor.generate.call(
    #         prompts=prompts,
    #         max_tokens=50,
    #         temperature=0.7
    #     )

    #     # Print results from rank 0
    #     if results[0] is not None:
    #         print("\n=== Generated Texts ===")
    #         for i, (prompt, result) in enumerate(zip(prompts, results[0])):
    #             print(f"\nPrompt {i+1}: {prompt}")
    #             print(f"Generated: {result}")
        
    #     # Cleanup
    #     await sglang_actor.cleanup.call()
        
    #     print("SGLang example completed successfully!")

    async def main():
        # Training example
        tp_actor, tp_mesh = await create_tp_actors()
        await tp_actor.setup.call()
        await tp_actor.step.call()
        await tp_actor.cleanup.call()
        
        # SGLang inference example
        # await run_sglang_example()
        sglang_actor = SGLangActor()

        print("SGLang actor created")
        prompts = [
            "Hello, my name is",
            "The meaning of life is",
            "In a world where AI"
        ]
        
        results = sglang_actor.generate(
            prompts=prompts,
            max_tokens=50,
            temperature=0.7
        )
        print(results)

        print("All examples completed successfully!")

    asyncio.run(main())

    # print("Monarch version:", monarch.__version__)

    # return {"message": "Monarch version: " + monarch.__version__}



if __name__ == "__main__":
    main.remote()
