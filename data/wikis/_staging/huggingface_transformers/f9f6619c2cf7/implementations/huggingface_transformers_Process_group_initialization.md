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
Usage pattern for torch.distributed.init_process_group within HuggingFace Transformers distributed training.

=== Description ===
Transformers automatically initializes the distributed process group when tensor parallelism is enabled, selecting the appropriate backend based on the device type. The initialization occurs in the tensor_parallel.py integration module, which handles the setup transparently when a model is loaded with a device_mesh or tp_plan parameter. The implementation extracts rank, local_rank, and world_size from environment variables set by the torchrun launcher, then calls PyTorch's init_process_group with the selected backend.

The backend selection logic maps device types to their optimal communication backends: NCCL for CUDA (NVIDIA GPUs), Gloo for CPU, CCL for Intel XPU, XCCL for newer Intel devices, and HCCL for Habana HPU.

=== Usage ===
In transformers workflows, the initialization happens automatically when you load a model with tensor parallelism enabled. Manual initialization is only needed if you're using custom distributed setups or need to initialize before model loading.

== Code Reference ==

=== Source Location ===
* '''Library:''' PyTorch
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/src/transformers/integrations/tensor_parallel.py:L65-88

=== Signature ===
<syntaxhighlight lang="python">
torch.distributed.init_process_group(
    backend: str,
    init_method: str = None,
    timeout: timedelta = default_timeout,
    world_size: int = None,
    rank: int = None,
    store: Store = None,
    group_name: str = "",
    pg_options: ProcessGroupOptions = None
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
import torch.distributed as dist
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| backend || str || Yes || Communication backend: "nccl", "gloo", "ccl", "xccl", "hccl"
|-
| rank || int || Yes || Unique identifier for this process (0 to world_size-1)
|-
| world_size || int || Yes || Total number of processes in the distributed group
|-
| init_method || str || No || URL specifying how to initialize the process group
|-
| timeout || timedelta || No || Timeout for collective operations
|-
| store || Store || No || Key-value store for rendezvous
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Process group is initialized globally, accessible via dist module
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import os
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

# Automatic initialization via transformers (recommended)
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)
# If not already initialized, transformers calls init_process_group internally

# Manual initialization (for custom setups)
if not torch.distributed.is_initialized():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Select backend based on device
    device_type = "cuda"
    backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "ccl", "hpu": "hccl"}
    backend = backend_map.get(device_type, "nccl")

    # Initialize process group
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )

    # Set device affinity
    if device_type != "cpu":
        torch.cuda.set_device(local_rank)

# Verify initialization
assert dist.is_initialized()
print(f"Rank {dist.get_rank()} of {dist.get_world_size()} initialized")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_Distributed_Init]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
