import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Suppose we have 4 devices for tensor parallel
tp_size = 4

# Create a mesh of shape (1, tp_size) with axes named (“dp”, “tp”)
mesh = jax.make_mesh((1, tp_size), ("dp", "tp"))  # dp axis has size 1, tp axis has size tp_size

# A toy linear function: y = x @ W^T
def f(x, W):
    return x @ W.T

# Partition specs:
# - We shard the weight W across the “tp” axis on its output dimension
# - Input x is replicated across tp (since dp=1)
# - Output is also sharded on the “tp” axis on the output dim
W = jnp.ones((8, 16))  # let's say input_dim=16, output_dim=8
x = jnp.ones((2, 16))  # batch size 2

# Annotate sharding
W_sharding = NamedSharding(mesh, P(None, "tp"))        # shard out_dim across “tp”
x_sharding = NamedSharding(mesh, P(None, None))         # replicate x across mesh
out_sharding = NamedSharding(mesh, P(None, "tp"))       # output sharded on “tp”

# Convert to sharded arrays
W_s = jax.make_array_from_single_device_arrays(W, W_sharding)
x_s = jax.make_array_from_single_device_arrays(x, x_sharding)

with mesh:
    y = jax.with_sharding_constraint(f(x_s, W_s), out_sharding)
    # do something with y
    print(y)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel

def setup_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def run():
    setup_dist()
    world_size = dist.get_world_size()
    # We assume world_size == tp_size, and dp_size = 1
    tp_size = world_size

    # Make a device mesh of shape (1, tp_size); we’ll only use “tp” axis
    mesh = init_device_mesh("cuda", (1, tp_size), mesh_dim_names=("dp", "tp"))

    # Build a small model
    model = MyLinear(in_features=16, out_features=8).cuda()

    # Because PyTorch’s parallelize_module only accepts a 1D mesh for TP,
    # we slice out the “tp” axis:
    tp_mesh = mesh["tp"]  # this gives a 1D DeviceMesh over the tp axis

    # Use ColwiseParallel style (shard output dim) for the linear layer
    model = parallelize_module(model, tp_mesh, {"linear": ColwiseParallel()})

    # Input x
    x = torch.ones((2, 16), device=torch.cuda.current_device())
    # Note: input will be replicated automatically by the DTensor system

    y = model(x)
    print(y)

if __name__ == "__main__":
    run()
