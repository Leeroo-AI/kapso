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
Usage pattern for torch.distributed.all_reduce for gradient synchronization within HuggingFace Transformers distributed training.

=== Description ===
Transformers uses PyTorch's all_reduce operation to manually synchronize gradients across data-parallel and context-parallel dimensions in 3D parallelism setups. The implementation in 3d_parallel_checks.py demonstrates the all_reduce_grads function, which iterates through model parameters and applies all_reduce to gradients across the dp_cp flattened mesh. The function handles both regular tensors and DTensors, converting DTensors to local tensors for cross-mesh communication, performing the all_reduce, then converting back.

When FSDP is used (use_ddp=True), it handles DP gradient synchronization automatically, so manual all_reduce is only needed for the CP dimension.

=== Usage ===
Call all_reduce on gradients after backward pass and before optimizer step when not using FSDP/DDP for gradient synchronization. In 3D parallelism, synchronize across the dp_cp mesh to aggregate gradients from all data-parallel and context-parallel ranks.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch (torch.distributed)
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/examples/pytorch/3d_parallel_checks.py:L558-584, L374

=== Signature ===
<syntaxhighlight lang="python">
torch.distributed.all_reduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup = None,
    async_op: bool = False
) -> Optional[Work]
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import torch.distributed as dist
from torch.distributed import ReduceOp
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tensor || torch.Tensor || Yes || Gradient tensor to synchronize (modified in-place)
|-
| op || ReduceOp || No || Reduction operation (SUM, AVG, MAX, MIN). Default: SUM
|-
| group || ProcessGroup || No || Process group for synchronization. Default: global group
|-
| async_op || bool || No || Whether to perform asynchronously. Default: False
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| work || Work or None || If async_op=True, returns Work handle for synchronization. Otherwise None
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Define 3D parallelism dimensions
tp_size = 2
dp_size = 2
cp_size = 2

# Create 3D device mesh
mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
world_mesh = DeviceMesh(
    device_type="cuda",
    mesh=mesh,
    mesh_dim_names=("dp", "tp", "cp")
)

# Extract submeshes
tp_mesh = world_mesh["tp"]
dp_mesh = world_mesh["dp"]
cp_mesh = world_mesh["cp"]

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)

# Training step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()

    # Manual gradient synchronization across DP and CP
    # (When not using FSDP/DDP)
    use_ddp = False  # Set to True if using FSDP

    if use_ddp:
        # FSDP handles DP, only sync CP manually
        sync_mesh = cp_mesh
    else:
        # Sync across both DP and CP
        sync_mesh = world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

    if sync_mesh.size() > 1:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if isinstance(param.grad, DTensor):
                    # Handle DTensor gradients
                    local_grad = param.grad.to_local()

                    # All-reduce with SUM
                    dist.all_reduce(
                        local_grad,
                        op=ReduceOp.SUM,
                        group=sync_mesh.get_group()
                    )

                    # Average by dividing by mesh size
                    local_grad = local_grad / sync_mesh.size()

                    # Convert back to DTensor
                    param.grad = DTensor.from_local(
                        local_grad,
                        device_mesh=param.grad.device_mesh,
                        placements=param.grad.placements
                    )
                else:
                    # Handle regular tensors
                    dist.all_reduce(
                        param.grad,
                        op=ReduceOp.AVG,  # Use AVG to avoid manual division
                        group=sync_mesh.get_group()
                    )

    optimizer.step()

# Complete implementation from 3d_parallel_checks.py
def all_reduce_grads(model, world_mesh, use_ddp):
    """All reduce gradients across dp_cp if applicable."""
    cp_mesh = world_mesh["cp"]

    if use_ddp:
        # FSDP/DDP takes care of DP syncing
        mesh = cp_mesh
    else:
        # Manual sync across both DP and CP
        mesh = world_mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

    if dist.is_initialized() and mesh.size() > 1:
        for name, param in model.named_parameters():
            if param.grad is not None:
                if isinstance(param.grad, DTensor):
                    # Workaround for cross-mesh communication with DTensor
                    local_grad = param.grad.to_local()

                    torch.distributed.all_reduce(
                        local_grad,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mesh.get_group()
                    )
                    local_grad = local_grad / mesh.size()

                    # Assign averaged grad back
                    param.grad = DTensor.from_local(
                        local_grad,
                        device_mesh=param.grad.device_mesh,
                        placements=param.grad.placements
                    )
                else:
                    # Handle regular tensors
                    torch.distributed.all_reduce(
                        param.grad,
                        op=torch.distributed.ReduceOp.AVG,
                        group=mesh.get_group()
                    )

# Usage in training loop
all_reduce_grads(model, world_mesh, use_ddp=False)

# Async all_reduce for overlapping computation and communication
work_handles = []
for param in model.parameters():
    if param.grad is not None:
        work = dist.all_reduce(
            param.grad,
            op=ReduceOp.AVG,
            group=dp_mesh.get_group(),
            async_op=True
        )
        work_handles.append(work)

# Do other computation here...

# Wait for all all_reduce operations to complete
for work in work_handles:
    work.wait()

optimizer.step()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Gradient_Synchronization]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
