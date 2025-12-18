{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|PyTorch Distributed|https://pytorch.org/docs/stable/distributed.html]]
|-
! Domains
| [[domain::Distributed_Computing]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Usage pattern for optimizer.step() within HuggingFace Transformers distributed training.

=== Description ===
Transformers uses standard PyTorch optimizers (AdamW, SGD, etc.) in distributed training, with the optimizer.step() method transparently handling parameter updates across different parallelism strategies. The implementation in 3d_parallel_checks.py demonstrates the typical workflow: zero gradients, forward pass, backward pass, gradient synchronization, gradient clipping, then optimizer.step().

The optimizer automatically works with DTensor parameters (from tensor parallelism) and FSDP-wrapped models (from data parallelism), updating only the locally available parameter shards or replicas. For gradient clipping, transformers uses either the FSDP's built-in clip_grad_norm_ method or a custom implementation that handles DTensors.

=== Usage ===
Call optimizer.step() after backward pass and gradient synchronization. Optionally perform gradient clipping before the step. The standard PyTorch optimizer API works without modification in distributed contexts.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch (torch.optim)
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/examples/pytorch/3d_parallel_checks.py:L307-409

=== Signature ===
<syntaxhighlight lang="python">
optimizer.step(
    closure: Callable = None
) -> Optional[float]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import torch.optim as optim
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| closure || Callable || No || Closure that re-evaluates model and returns loss (for some optimizers)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| loss || float or None || Loss value from closure if provided, otherwise None
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForCausalLM

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Define 3D parallelism
tp_size = 2
dp_size = 4
cp_size = 1

# Create 3D device mesh
mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
world_mesh = DeviceMesh(
    device_type="cuda",
    mesh=mesh,
    mesh_dim_names=("dp", "tp", "cp")
)

tp_mesh = world_mesh["tp"]
dp_mesh = world_mesh["dp"]

# Load model with tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)

# Wrap with FSDP for data parallelism
use_fsdp = dp_mesh.size() > 1
if use_fsdp:
    model = FSDP(
        model,
        device_mesh=dp_mesh,
        sharding_strategy=ShardingStrategy.NO_SHARD
    )

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

# Training loop
for batch in dataloader:
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss

    # Backward pass
    loss.backward()

    # Gradient synchronization (manual if not using FSDP)
    if not use_fsdp:
        # all_reduce_grads(model, world_mesh, use_ddp=False)
        pass

    # Gradient clipping
    if hasattr(model, "clip_grad_norm_"):
        # FSDP provides built-in clipping
        gradnorm = model.clip_grad_norm_(max_norm=1.0, norm_type=2.0)
    else:
        # Manual clipping for non-FSDP models
        gradnorm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            norm_type=2.0
        )

    # Optimizer step
    optimizer.step()

    if rank == 0:
        print(f"Step completed. Loss: {loss.item():.4f}, Gradnorm: {gradnorm:.4f}")

# Custom gradient clipping for DTensor (from 3d_parallel_checks.py)
from collections.abc import Iterable
from torch.distributed.tensor import DTensor

def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """Clip gradient norm of an iterable of parameters."""
    # Filter parameters with gradients
    parameters = [p for p in parameters if p.grad is not None]
    assert len(parameters) > 0, "No parameters with gradients found"

    # Calculate total norm
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type
        )

    # Convert DTensor to local tensor if needed
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    return total_norm

# Usage with custom clipping
gradnorm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
optimizer.step()

# With learning rate scheduler
from transformers import get_linear_schedule_with_warmup

num_training_steps = 10000
num_warmup_steps = 100

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step
    optimizer.step()

    # Learning rate scheduler step
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    if rank == 0 and step % 100 == 0:
        print(f"Step {step}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

# Verification: Check parameters updated
initial_params = {name: param.clone() for name, param in model.named_parameters()}

# Training step
optimizer.zero_grad()
outputs = model(**batch)
loss = outputs.loss
loss.backward()
optimizer.step()

# Verify parameters changed
for name, param in model.named_parameters():
    param_changed = not torch.allclose(
        param if not isinstance(param, DTensor) else param.to_local(),
        initial_params[name] if not isinstance(initial_params[name], DTensor)
        else initial_params[name].to_local()
    )
    assert param_changed, f"Parameter {name} was not updated!"

print(f"Rank {rank}: All parameters updated successfully")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Distributed_Optimizer_Step]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
