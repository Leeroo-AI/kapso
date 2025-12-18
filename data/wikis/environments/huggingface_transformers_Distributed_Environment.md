# Environment: huggingface_transformers_Distributed_Environment

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
* [[source::Doc|Distributed Training|https://huggingface.co/docs/transformers/perf_train_gpu_many]]
|-
! Domains
| [[domain::Distributed_Training]], [[domain::Parallelism]], [[domain::High_Performance]]
|-
! Last Updated
| [[last_updated::2025-12-18 14:00 GMT]]
|}

== Overview ==
Multi-GPU environment with PyTorch 2.2+, NCCL backend, and support for Tensor Parallelism (TP), Data Parallelism (DP/FSDP), and Context Parallelism (CP).

=== Description ===
This environment provides the advanced distributed training stack for 3D parallelism. It combines Tensor Parallelism (model layer sharding across GPUs), Data Parallelism (batch sharding via FSDP or DDP), and Context Parallelism (sequence length sharding). The environment requires multiple GPUs with high-bandwidth interconnect (NVLink/NVSwitch recommended) and the NCCL communication backend.

=== Usage ===
Use this environment for training **large models** that don't fit on a single GPU, or for **scaling** training across multiple GPUs/nodes. Required when using `device_mesh`, `tp_plan="auto"`, FSDP, or DeepSpeed.

== System Requirements ==

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Linux || Required for NCCL
|-
| Python || Python >= 3.10 || Required by transformers
|-
| Hardware || Multiple NVIDIA GPUs || NVLink recommended for TP
|-
| GPU Memory || 16GB+ per GPU || A100/H100 recommended
|-
| Network || High-bandwidth || InfiniBand for multi-node
|}

== Dependencies ==

=== System Packages ===
* `cuda-toolkit` >= 11.8 - CUDA runtime
* `cudnn` >= 8.6 - Optimized CUDA operations
* `nccl` >= 2.14 - NVIDIA Collective Communications Library

=== Python Packages ===
* `transformers` (this package)
* `torch` >= 2.2 - **Required** for DTensor/DeviceMesh
* `accelerate` >= 1.1.0 - For FSDP configuration
* `datasets` >= 2.15.0 - For data loading

=== Optional Dependencies ===
* `deepspeed` >= 0.9.3 - For ZeRO optimization
* `wandb` - For distributed experiment tracking
* `torch-xla` - For TPU support

== Credentials ==
* `HF_TOKEN`: For private models
* `WANDB_API_KEY`: For experiment tracking

== Quick Install ==

<syntaxhighlight lang="bash">
# Basic distributed environment
pip install transformers torch accelerate datasets

# With DeepSpeed
pip install transformers torch accelerate datasets deepspeed

# With experiment tracking
pip install transformers torch accelerate datasets wandb
</syntaxhighlight>

== Code Evidence ==

3D parallelism setup from `examples/3D_parallel.py:L74-106`:

<syntaxhighlight lang="python">
tp_size = int(os.environ.get("TP_SIZE", "1"))
dp_size = int(os.environ.get("DP_SIZE", "1"))
cp_size = int(os.environ.get("CP_SIZE", "1"))

# Initialize distributed environment
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

assert world_size == tp_size * dp_size * cp_size, (
    f"World size ({world_size}) must equal TP * DP * CP"
)

mesh = torch.arange(world_size).reshape(dp_size, tp_size, cp_size)
world_mesh = DeviceMesh(device_type="cuda", mesh=mesh,
                        mesh_dim_names=("dp", "tp", "cp"))
</syntaxhighlight>

Tensor Parallel model loading from `modeling_utils.py`:

<syntaxhighlight lang="python">
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_mesh=tp_mesh,  # DeviceMesh for tensor parallelism
    tp_plan="auto",       # Automatic TP sharding plan
    dtype=torch.bfloat16,
)
</syntaxhighlight>

FSDP wrapping from `examples/3D_parallel.py:L151-153`:

<syntaxhighlight lang="python">
if dp_mesh.size() > 1:
    model = FSDP(model, device_mesh=dp_mesh,
                 sharding_strategy=ShardingStrategy.NO_SHARD)
</syntaxhighlight>

== Common Errors ==

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `RuntimeError: NCCL error` || GPU communication failure || Check NCCL version, GPU visibility
|-
|| `AssertionError: World size must equal TP * DP * CP` || Mismatch in parallelism config || Adjust TP_SIZE, DP_SIZE, CP_SIZE
|-
|| `RuntimeError: Expected all tensors on same device` || Device placement error || Use proper device_mesh configuration
|-
|| `OutOfMemoryError` || Model too large for available GPUs || Increase TP_SIZE, enable CPU offload
|-
|| `NCCL timeout` || Slow inter-GPU communication || Increase timeout, check network
|}

== Compatibility Notes ==

* **PyTorch >= 2.2:** Required for native DTensor/DeviceMesh support
* **NCCL Backend:** Required for NVIDIA GPUs; `gloo` for CPU
* **NVLink/NVSwitch:** Highly recommended for TP (high bandwidth needed)
* **FSDP Version:** FSDP2 (PyTorch 2.4+) recommended for large models
* **Context Parallelism:** Experimental feature in PyTorch, requires Flash Attention
* **Multi-node:** Requires proper network configuration (InfiniBand recommended)

== Environment Variables ==

<syntaxhighlight lang="bash">
# Required for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RANK=0
export WORLD_SIZE=8
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Parallelism configuration
export TP_SIZE=2  # Tensor Parallelism degree
export DP_SIZE=2  # Data Parallelism degree
export CP_SIZE=2  # Context Parallelism degree
</syntaxhighlight>

== Launch Commands ==

<syntaxhighlight lang="bash">
# Single node, 4 GPUs with TP=2, DP=2
TP_SIZE=2 DP_SIZE=2 torchrun --nproc_per_node=4 train.py

# Single node, 8 GPUs with 3D parallelism
DP_SIZE=2 CP_SIZE=2 TP_SIZE=2 torchrun --nproc_per_node=8 train.py

# Multi-node (node 0)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
         --master_addr=node0 --master_port=29500 train.py

# With accelerate
accelerate launch --multi_gpu --num_processes=4 train.py
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Implementation:huggingface_transformers_Process_group_initialization]]
* [[requires_env::Implementation:huggingface_transformers_TensorParallel_from_pretrained]]
* [[requires_env::Implementation:huggingface_transformers_FSDP_wrapping]]
* [[requires_env::Implementation:huggingface_transformers_DistributedSampler_usage]]
* [[requires_env::Implementation:huggingface_transformers_Context_parallel_execution]]
* [[requires_env::Implementation:huggingface_transformers_AllReduce_gradients]]
* [[requires_env::Implementation:huggingface_transformers_Optimizer_step]]
* [[requires_env::Implementation:huggingface_transformers_DCP_save]]
