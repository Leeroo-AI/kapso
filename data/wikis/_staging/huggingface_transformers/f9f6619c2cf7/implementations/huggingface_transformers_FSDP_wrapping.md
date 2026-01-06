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
Usage pattern for FSDP (Fully Sharded Data Parallel) within HuggingFace Transformers distributed training.

=== Description ===
Transformers applies FSDP wrapping to models in 3D parallelism setups to enable data-parallel gradient synchronization across the data-parallel dimension. The wrapping occurs after tensor-parallel model loading, converting the TP-sharded model into an FSDP module that manages gradient synchronization automatically. In the 3d_parallel_checks.py example, FSDP is applied with NO_SHARD strategy when dp_size > 1, effectively creating DDP-style replication while maintaining FSDP's interface.

The device_mesh parameter specifies the data-parallel submesh, ensuring FSDP only synchronizes gradients across the intended data-parallel ranks, not across tensor-parallel or context-parallel dimensions.

=== Usage ===
Wrap your tensor-parallel model with FSDP after loading and before training when dp_size > 1. Use NO_SHARD strategy for simple gradient synchronization or FULL_SHARD for memory-efficient training of very large models.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch (torch.distributed.fsdp)
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/examples/pytorch/3d_parallel_checks.py:L182-192

=== Signature ===
<syntaxhighlight lang="python">
FSDP(
    module: nn.Module,
    process_group: ProcessGroup = None,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    cpu_offload: CPUOffload = None,
    auto_wrap_policy: Callable = None,
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE,
    mixed_precision: MixedPrecision = None,
    device_id: torch.device = None,
    device_mesh: DeviceMesh = None,
    **kwargs
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.device_mesh import DeviceMesh
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| module || nn.Module || Yes || Model to wrap (can already be tensor-parallel)
|-
| sharding_strategy || ShardingStrategy || No || FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, or HYBRID_SHARD
|-
| device_mesh || DeviceMesh || No || Device mesh for data-parallel dimension
|-
| process_group || ProcessGroup || No || Process group for synchronization (derived from device_mesh)
|-
| auto_wrap_policy || Callable || No || Policy for automatic submodule wrapping
|-
| mixed_precision || MixedPrecision || No || Configuration for mixed-precision training
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| wrapped_model || FSDP || Model wrapped for data-parallel gradient synchronization
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForCausalLM

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Define 3D parallelism dimensions
tp_size = 2
dp_size = 4
cp_size = 1
assert world_size == tp_size * dp_size * cp_size

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

# Load model with tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)

# Wrap with FSDP for data parallelism
if dp_mesh.size() > 1:
    model = FSDP(
        model,
        device_mesh=dp_mesh,
        sharding_strategy=ShardingStrategy.NO_SHARD  # DDP-style
    )
    print(f"Rank {rank}: Model wrapped with FSDP")

# Alternative: Full sharding for memory efficiency
model = FSDP(
    model,
    device_mesh=dp_mesh,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)

# Training loop with automatic gradient synchronization
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()  # FSDP automatically reduces gradients across dp_mesh

    optimizer.step()

# FSDP2 API (newer, more flexible)
from torch.distributed._composable.fsdp import fully_shard

# Wrap each transformer layer individually
for transformer_block in model.model.layers:
    fully_shard(
        transformer_block,
        mesh=dp_mesh,
        reshard_after_forward=False
    )

# Wrap the entire model
fully_shard(model.model, mesh=dp_mesh, reshard_after_forward=False)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Data_Parallelism_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
