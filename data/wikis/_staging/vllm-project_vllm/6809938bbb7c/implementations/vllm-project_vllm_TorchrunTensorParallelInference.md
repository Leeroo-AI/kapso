{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Offline Inference]], [[domain::Tensor Parallelism]], [[domain::Pipeline Parallelism]], [[domain::Distributed Inference]], [[domain::Torchrun]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates tensor-parallel and pipeline-parallel inference with vLLM using torchrun as an external launcher for distributed execution.

=== Description ===
This example shows how to use PyTorch's torchrun utility to launch vLLM with tensor parallelism (TP) and/or pipeline parallelism (PP), enabling inference with large models that require multiple GPUs. Instead of vLLM's internal process spawning, torchrun manages the distributed processes, providing better integration with cluster schedulers and standard PyTorch distributed tooling.

The script demonstrates the <code>distributed_executor_backend="external_launcher"</code> mode where torchrun spawns all processes, and vLLM discovers the distributed configuration through environment variables. All ranks execute the same code and produce identical outputs through synchronized inference.

This pattern is particularly useful for HPC environments, Kubernetes deployments, or workflows that already use torchrun for other PyTorch distributed tasks.

=== Usage ===
Use this approach when:
* Running in HPC clusters with SLURM or PBS that integrate well with torchrun
* Deploying to Kubernetes with PyTorch training operators
* Needing fine-grained control over process placement and GPU assignment
* Integrating vLLM into existing PyTorch distributed workflows
* Working with very large models requiring tensor or pipeline parallelism
* Debugging distributed issues with standard PyTorch tools

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_example.py examples/offline_inference/torchrun_example.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Basic tensor parallelism (TP=2)
torchrun --nproc-per-node=2 \
    examples/offline_inference/torchrun_example.py

# Tensor + Pipeline parallelism (TP=2, PP=2, total 4 GPUs)
torchrun --nproc-per-node=4 \
    examples/offline_inference/torchrun_example.py

# Multi-node inference (2 nodes, 8 GPUs each, TP=16)
# On master node:
torchrun --nproc-per-node=8 \
    --nnodes=2 \
    --node-rank=0 \
    --master-addr=MASTER_IP \
    --master-port=29500 \
    examples/offline_inference/torchrun_example.py

# On worker node:
torchrun --nproc-per-node=8 \
    --nnodes=2 \
    --node-rank=1 \
    --master-addr=MASTER_IP \
    --master-port=29500 \
    examples/offline_inference/torchrun_example.py
</syntaxhighlight>

== Key Concepts ==

=== External Launcher Mode ===
The <code>distributed_executor_backend="external_launcher"</code> setting:
* Tells vLLM that processes are pre-spawned by an external tool (torchrun)
* Each process discovers its rank and world size from environment variables:
  * <code>RANK</code>, <code>WORLD_SIZE</code>, <code>LOCAL_RANK</code>
  * <code>MASTER_ADDR</code>, <code>MASTER_PORT</code>
* vLLM creates exactly one worker per process (no internal spawning)
* Enables integration with standard PyTorch distributed launchers

=== Tensor Parallelism with Torchrun ===
Configuration in the example:
<syntaxhighlight lang="python">
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    distributed_executor_backend="external_launcher",
    max_model_len=32768,
    seed=1,
)
</syntaxhighlight>

Requirements:
* <code>nproc_per_node</code> must equal <code>tensor_parallel_size * pipeline_parallel_size</code>
* All ranks run the same Python script
* Synchronization ensures identical outputs across all ranks

=== Deterministic Execution ===
The example sets <code>seed=1</code> to ensure:
* All ranks use the same random seed
* Sampling produces identical results across ranks
* Outputs are deterministic for the same inputs
* Critical for correctness verification in distributed setups

=== Output Handling ===
Only rank 0 prints outputs to avoid duplicate output:
<syntaxhighlight lang="python">
import torch.distributed as dist

outputs = llm.generate(prompts, sampling_params)

if dist.get_rank() == 0:
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
</syntaxhighlight>

All ranks compute the same outputs, but only one prints to avoid clutter.

== Usage Examples ==

=== Basic Tensor Parallel Inference ===
<syntaxhighlight lang="python">
import torch.distributed as dist
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create LLM with external launcher
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    distributed_executor_backend="external_launcher",
    max_model_len=32768,
    seed=1,
)

# All ranks generate outputs
outputs = llm.generate(prompts, sampling_params)

# Only rank 0 prints
if dist.get_rank() == 0:
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")
</syntaxhighlight>

<syntaxhighlight lang="bash">
# Run with torchrun (4 processes for TP=2, PP=2)
torchrun --nproc-per-node=4 my_script.py
</syntaxhighlight>

=== Large Model Inference ===
<syntaxhighlight lang="python">
# 70B model requiring TP=4
llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    distributed_executor_backend="external_launcher",
    dtype="bfloat16",
    max_model_len=8192,
    seed=42,
)

# Launch with 4 GPUs
# torchrun --nproc-per-node=4 script.py
</syntaxhighlight>

=== Accessing Distributed State ===
<syntaxhighlight lang="python">
from vllm.distributed.parallel_state import get_world_group
import torch.distributed as dist

# Get CPU group (GLOO backend) for control messages
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)

if torch_rank == 0:
    print("I'm the leader, saving results...")
    save_results(outputs, "results.json")

# Get device group (NCCL backend) for GPU operations
device_group = get_world_group().device_group

# Synchronize all ranks
dist.barrier(group=cpu_group)
</syntaxhighlight>

=== Direct Model Access ===
<syntaxhighlight lang="python">
# Access the model directly for inspection or modification
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

# Inspect model structure
if dist.get_rank() == 0:
    print(f"Model type: {type(model)}")
    print(f"Number of layers: {len(model.layers)}")

    # Check sharding
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
</syntaxhighlight>

=== SLURM Integration ===
<syntaxhighlight lang="bash">
#!/bin/bash
#SBATCH --job-name=vllm-inference
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8

# SLURM automatically sets up environment for torchrun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=8 \
    --node-rank=$SLURM_NODEID \
    examples/offline_inference/torchrun_example.py
</syntaxhighlight>

=== Kubernetes with PyTorch Operator ===
<syntaxhighlight lang="yaml">
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: vllm-inference
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: vllm/vllm-openai:latest
            command:
            - python
            - examples/offline_inference/torchrun_example.py
            resources:
              limits:
                nvidia.com/gpu: 4
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: pytorch
            image: vllm/vllm-openai:latest
            command:
            - python
            - examples/offline_inference/torchrun_example.py
            resources:
              limits:
                nvidia.com/gpu: 4
</syntaxhighlight>

== Advanced Tips from the Example ==

The example includes several advanced tips in comments:

=== Using CPU Process Group ===
<syntaxhighlight lang="python">
from vllm.distributed.parallel_state import get_world_group
import torch.distributed as dist

# CPU group uses GLOO backend (works for CPU tensors and metadata)
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)

if torch_rank == 0:
    # Operations that should only run on one rank
    results = collect_results(outputs)
    save_to_disk(results, "results.json")
    log_metrics(results)
</syntaxhighlight>

=== Using Device Process Group ===
<syntaxhighlight lang="python">
# Device group uses NCCL backend (for GPU tensors)
device_group = get_world_group().device_group

# Custom all-reduce for aggregating results
import torch
local_metrics = compute_metrics(outputs)
global_metrics = torch.zeros_like(local_metrics)
dist.all_reduce(global_metrics, op=dist.ReduceOp.SUM, group=device_group)
</syntaxhighlight>

=== Direct Model Manipulation ===
<syntaxhighlight lang="python">
# Access the model for advanced use cases
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

# Example: Inspect activation patterns
hooks = []
def hook_fn(module, input, output):
    print(f"Layer {module}: output shape {output.shape}")

for layer in model.layers:
    hooks.append(layer.register_forward_hook(hook_fn))

# Run inference with hooks active
outputs = llm.generate(prompts, sampling_params)

# Clean up hooks
for hook in hooks:
    hook.remove()
</syntaxhighlight>

== Performance Considerations ==

=== Initialization Overhead ===
* Torchrun adds ~5-10 seconds for process group initialization
* Each process loads its model shard independently
* Total initialization similar to vLLM's internal launcher
* Better integration with cluster schedulers may reduce queue time

=== Communication Patterns ===
* Tensor parallelism requires all-reduce on every forward pass
* Pipeline parallelism has point-to-point communication between stages
* Network bandwidth critical for multi-node setups (use InfiniBand/NVLink)
* Latency increases with TP/PP size due to synchronization

=== Scalability ===
* Single-node: Excellent (NVLink provides high bandwidth)
* Multi-node: Good (requires high-bandwidth interconnect)
* TP efficiency: >85% up to TP=8, degrades beyond TP=16
* PP efficiency: ~90% for PP=2-4, better for very large models

== Debugging Distributed Issues ==

=== Rank Output ===
<syntaxhighlight lang="python">
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"[Rank {rank}/{world_size}] Starting inference...")

outputs = llm.generate(prompts, sampling_params)

print(f"[Rank {rank}/{world_size}] Generated {len(outputs)} outputs")
</syntaxhighlight>

=== Environment Inspection ===
<syntaxhighlight lang="python">
import os

print(f"RANK: {os.environ.get('RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
</syntaxhighlight>

=== Synchronization Barriers ===
<syntaxhighlight lang="python">
import torch.distributed as dist
from vllm.distributed.parallel_state import get_world_group

cpu_group = get_world_group().cpu_group

print(f"[Rank {dist.get_rank()}] Before barrier")
dist.barrier(group=cpu_group)
print(f"[Rank {dist.get_rank()}] After barrier")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_TorchrunDataParallelInference]]
* [[related::Concept:vllm-project_vllm_Tensor_Parallelism]]
* [[related::Concept:vllm-project_vllm_Pipeline_Parallelism]]
* [[related::Concept:vllm-project_vllm_Distributed_Inference]]
