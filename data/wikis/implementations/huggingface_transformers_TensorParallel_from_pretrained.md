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
Usage pattern for AutoModelForCausalLM.from_pretrained with tensor parallelism in HuggingFace Transformers.

=== Description ===
Transformers extends the standard from_pretrained method to support automatic tensor-parallel model loading via the device_mesh and tp_plan parameters. When these parameters are provided, the model weights are automatically sharded across the specified devices according to the parallelism plan. The "auto" tp_plan triggers automatic analysis of the model architecture to determine optimal sharding strategies for each layer.

The implementation first initializes distributed communication if not already done, then loads the model configuration and weights while applying the specified tensor-parallel transformations. Each rank loads only its assigned weight shards, significantly reducing per-device memory requirements for large models.

=== Usage ===
Use this when loading large models for distributed training or inference with tensor parallelism. The device_mesh defines the parallel topology, and tp_plan="auto" works for most standard architectures. Custom tp_plan dictionaries provide fine-grained control over sharding strategies.

== Code Reference ==

=== Source Location ===
* '''Library:''' HuggingFace Transformers
* '''Transformers Usage:''' /tmp/praxium_repo_d5p6fp4d/src/transformers/modeling_utils.py:L3563-4200

=== Signature ===
<syntaxhighlight lang="python">
AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path: str | os.PathLike,
    *model_args,
    config: PreTrainedConfig = None,
    cache_dir: str | os.PathLike = None,
    device_mesh: DeviceMesh = None,
    tp_plan: str | dict = None,
    dtype: torch.dtype = None,
    **kwargs
) -> PreTrainedModel
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from torch.distributed.device_mesh import DeviceMesh
import torch
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| pretrained_model_name_or_path || str || Yes || Model identifier from Hub or local path
|-
| device_mesh || DeviceMesh || No || Defines tensor-parallel device topology
|-
| tp_plan || str or dict || No || "auto" for automatic or dict for custom sharding strategy
|-
| dtype || torch.dtype || No || Data type for model parameters (e.g., torch.bfloat16)
|-
| config || PreTrainedConfig || No || Custom model configuration
|-
| cache_dir || str || No || Directory for caching downloaded models
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Model with parameters sharded as DTensors across device_mesh
|}

== Usage Examples ==

=== Example in Transformers Context ===
<syntaxhighlight lang="python">
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create device mesh for tensor parallelism
# For 4 GPUs: [0, 1, 2, 3]
tp_mesh = DeviceMesh(device_type="cuda", mesh=torch.arange(world_size))

# Load model with automatic tensor parallelism
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",  # Automatic sharding strategy
    dtype=torch.bfloat16
)

# Verify model is sharded
for name, param in model.named_parameters():
    if hasattr(param, 'placements'):
        print(f"{name}: {param.shape}, placements={param.placements}")

# For 3D parallelism (TP + DP + CP)
tp_size = 2
dp_size = 2
cp_size = 2
world_size = tp_size * dp_size * cp_size  # 8 total ranks

# Create 3D mesh: (dp, tp, cp)
mesh_3d = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
world_mesh = DeviceMesh(
    device_type="cuda",
    mesh=mesh_3d,
    mesh_dim_names=("dp", "tp", "cp")
)

# Extract tensor-parallel submesh
tp_mesh = world_mesh["tp"]

# Load with TP only (DP and CP applied separately)
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan="auto",
    dtype=torch.bfloat16
)

# Custom tp_plan for fine-grained control
custom_tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
}

model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-1.7B",
    device_mesh=tp_mesh,
    tp_plan=custom_tp_plan,
    dtype=torch.bfloat16
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_transformers_TP_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_transformers_Distributed_Environment]]
