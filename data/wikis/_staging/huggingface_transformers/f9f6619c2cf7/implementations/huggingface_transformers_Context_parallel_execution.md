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
Usage pattern for context_parallel context manager within HuggingFace Transformers distributed training.

=== Description ===
Transformers uses PyTorch's experimental context_parallel context manager to enable sequence parallelism in long-context training scenarios. The context manager is applied around the model's forward and backward passes, automatically sharding specified input buffers (input_ids, labels, position_ids) along their sequence dimensions. The implementation in 3d_parallel_checks.py demonstrates wrapping the forward pass with context_parallel when cp_mesh.size() > 1, using nullcontext otherwise.

The buffer_seq_dims parameter indicates which dimension of each buffer represents the sequence length (typically dimension 1 for [batch, sequence, ...] tensors). The context manager handles the ring attention communication internally, allowing the model to compute with sequences longer than single-device memory.

=== Usage ===
Wrap your model's forward and backward computation within the context_parallel context manager when cp_size > 1. Specify all tensors that should be sharded along the sequence dimension in the buffers list, and provide their sequence dimension indices in buffer_seq_dims.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch (torch.distributed.tensor.experimental)
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/examples/pytorch/3d_parallel_checks.py:L50-51, L340-370

=== Signature ===
<syntaxhighlight lang="python">
context_parallel(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    buffer_seq_dims: list[int] = None
) -> ContextManager
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.device_mesh import DeviceMesh
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| mesh || DeviceMesh || Yes || Context-parallel device mesh
|-
| buffers || list[torch.Tensor] || Yes || Tensors to shard along sequence dimension
|-
| buffer_seq_dims || list[int] || No || Sequence dimension index for each buffer (defaults based on tensor shape)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| context || ContextManager || Context manager that handles sequence sharding and ring attention
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel
from contextlib import nullcontext
from transformers import AutoModelForCausalLM

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Define 3D parallelism dimensions
tp_size = 2
dp_size = 2
cp_size = 2
seq_len = 4096

# Create 3D device mesh
mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
world_mesh = DeviceMesh(
    device_type="cuda",
    mesh=mesh,
    mesh_dim_names=("dp", "tp", "cp")
)
tp_mesh = world_mesh["tp"]
cp_mesh = world_mesh["cp"]

# Load model with tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)

# Training loop with context parallelism
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    optimizer.zero_grad()

    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # Add position_ids
    batch_size = batch["input_ids"].shape[0]
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    batch["position_ids"] = position_ids

    # Use context parallelism if cp_size > 1
    sdpa_backend = SDPBackend.FLASH_ATTENTION  # Required for CP

    with sdpa_kernel(sdpa_backend):
        cp_context = (
            nullcontext()
            if cp_mesh.size() == 1
            else context_parallel(
                cp_mesh,
                buffers=[
                    batch["input_ids"],
                    batch["labels"],
                    batch["position_ids"],
                ],
                buffer_seq_dims=[1, 1, 1]  # Sequence is dimension 1
            )
        )

        with cp_context:
            # Inside this context, tensors are sharded along sequence dimension
            # Each CP rank holds seq_len/cp_size tokens

            # Pop labels for separate loss computation
            labels = batch.pop("labels")

            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits  # Shape: [batch, seq_len/cp_size, vocab_size]

            # Compute loss
            loss = model.loss_function(
                logits=logits,
                labels=None,
                shift_labels=labels,
                vocab_size=model.config.vocab_size
            )

            # Backward pass (ring attention gradients automatically handled)
            loss.backward()

    # All-reduce gradients across DP and CP if needed
    # (handled by FSDP or manual all_reduce)

    optimizer.step()

# Verify sequence sharding across CP ranks
print(f"CP rank {cp_mesh.get_local_rank()}: "
      f"Processing sequence chunk of size {seq_len // cp_mesh.size()}")

# Example with manual buffer preparation
full_seq_len = 8192
local_seq_len = full_seq_len // cp_mesh.size()
cp_rank = cp_mesh.get_local_rank()

# Each rank prepares its sequence chunk
start_idx = cp_rank * local_seq_len
end_idx = start_idx + local_seq_len

input_ids_full = torch.randint(0, 50000, (2, full_seq_len), device=device)
input_ids_local = input_ids_full[:, start_idx:end_idx]

position_ids_local = torch.arange(start_idx, end_idx, device=device)
position_ids_local = position_ids_local.unsqueeze(0).expand(2, -1)

# Use in context_parallel
with context_parallel(
    cp_mesh,
    buffers=[input_ids_local, position_ids_local],
    buffer_seq_dims=[1, 1]
):
    outputs = model(input_ids=input_ids_local, position_ids=position_ids_local)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Context_Parallelism]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
