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
Usage pattern for DistributedSampler within HuggingFace Transformers distributed training.

=== Description ===
Transformers uses PyTorch's DistributedSampler to partition datasets across data-parallel ranks in 3D parallelism setups. The sampler is instantiated with num_replicas set to the data-parallel mesh size and rank set to the local rank within the data-parallel dimension. This ensures each DP rank processes unique data while TP and CP ranks process identical data.

The implementation in 3d_parallel_checks.py demonstrates creating the sampler with shuffle=False (since the dataset is pre-shuffled) and passing it to the DataLoader to control index assignment across ranks.

=== Usage ===
Create a DistributedSampler when initializing your DataLoader in data-parallel training scenarios. Use the data-parallel mesh's size and local rank, not the global world size/rank, to ensure proper partitioning in multi-dimensional parallelism.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch (torch.utils.data.distributed)
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/examples/pytorch/3d_parallel_checks.py:L220-250

=== Signature ===
<syntaxhighlight lang="python">
DistributedSampler(
    dataset: Dataset,
    num_replicas: int = None,
    rank: int = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| dataset || Dataset || Yes || The dataset to sample from
|-
| num_replicas || int || No || Number of data-parallel processes (defaults to world_size)
|-
| rank || int || No || Rank of current process in data-parallel group
|-
| shuffle || bool || No || Whether to shuffle indices (default: True)
|-
| seed || int || No || Random seed for shuffling (default: 0)
|-
| drop_last || bool || No || Drop tail samples to make evenly divisible (default: False)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| sampler || DistributedSampler || Sampler that yields indices for this rank's partition
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Define 3D parallelism dimensions
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
dp_mesh = world_mesh["dp"]

# Load and tokenize dataset
dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=1024
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create DistributedSampler for data parallelism
sampler = DistributedSampler(
    tokenized_dataset,
    num_replicas=dp_mesh.size(),  # Number of DP ranks
    rank=dp_mesh.get_local_rank(),  # This rank's position in DP dimension
    shuffle=False  # Dataset already shuffled
)

# Calculate local batch size
global_batch_size = 32
local_batch_size = global_batch_size // dp_mesh.size()

# Create DataLoader with sampler
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=local_batch_size,
    sampler=sampler,
    shuffle=False,  # Must be False when using sampler
    collate_fn=lambda x: x  # Custom collate function
)

# Training loop
for epoch in range(num_epochs):
    # Important: Set epoch for proper shuffling across epochs
    sampler.set_epoch(epoch)

    for batch in dataloader:
        # Each DP rank processes different data
        # TP and CP ranks process identical data
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Verify data partitioning across DP ranks
if dp_mesh.size() > 1:
    # Get first batch indices from each DP rank
    local_indices = next(iter(dataloader))['input_ids'][0][:10]

    # Gather from all DP ranks
    gathered = [torch.zeros_like(local_indices) for _ in range(dp_mesh.size())]
    dist.all_gather(gathered, local_indices, group=dp_mesh.get_group())

    if rank == 0:
        # Verify no overlap
        for i in range(len(gathered)):
            for j in range(i+1, len(gathered)):
                assert not torch.equal(gathered[i], gathered[j]), \
                    f"DP ranks {i} and {j} have overlapping data!"
        print("Data partitioning verified: No overlap across DP ranks")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Distributed_Dataset]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
