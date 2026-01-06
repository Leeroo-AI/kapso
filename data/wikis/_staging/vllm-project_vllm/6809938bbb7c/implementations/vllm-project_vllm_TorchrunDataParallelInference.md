{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Offline Inference]], [[domain::Data Parallelism]], [[domain::Distributed Inference]], [[domain::Torchrun]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates data-parallel inference with vLLM using torchrun, enabling horizontal scaling across multiple GPUs or nodes for higher throughput.

=== Description ===
This example shows how to use PyTorch's torchrun launcher to run vLLM in data-parallel mode, where multiple independent model instances process different subsets of prompts simultaneously. Unlike tensor parallelism (which splits a single model across GPUs), data parallelism replicates the entire model on each GPU/group and distributes the workload.

The script uses <code>distributed_executor_backend="external_launcher"</code> to integrate with torchrun's process management, automatically discovering ranks and world size. Each data-parallel rank processes a subset of prompts determined by its rank ID, enabling efficient load distribution for batch processing workloads.

This approach supports combining data parallelism with tensor/pipeline parallelism and expert parallelism for flexible multi-dimensional scaling.

=== Usage ===
Use this approach when:
* You need higher aggregate throughput than a single model instance can provide
* Processing large batches of independent requests (batch inference, evaluation)
* Model fits on a single GPU but you want to use multiple GPUs for speed
* Scaling out across multiple nodes for massive throughput
* Combining with tensor parallelism for large models (TP within groups, DP across groups)
* Serving multiple users with isolated model instances (data privacy)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_dp_example.py examples/offline_inference/torchrun_dp_example.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Basic data parallelism with 2 GPUs
torchrun --nproc-per-node=2 \
    examples/offline_inference/torchrun_dp_example.py

# Data parallelism with 8 GPUs (4 DP ranks, TP=2 within each)
torchrun --nproc-per-node=8 \
    examples/offline_inference/torchrun_dp_example.py \
    --tp-size=2 --pp-size=1 --dp-size=4

# With expert parallelism for MoE models
torchrun --nproc-per-node=8 \
    examples/offline_inference/torchrun_dp_example.py \
    --model microsoft/Phi-mini-MoE-instruct \
    --tp-size=2 --dp-size=4 --enable-ep

# Multi-node data parallelism (run on each node)
torchrun --nproc-per-node=8 \
    --nnodes=4 \
    --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    examples/offline_inference/torchrun_dp_example.py \
    --dp-size=32 --tp-size=1
</syntaxhighlight>

== Key Concepts ==

=== Data Parallelism (DP) ===
In data parallelism:
* Each DP rank runs a complete model instance
* Prompts are partitioned across ranks (external load balancing)
* Each rank independently generates outputs
* No communication between ranks during inference (embarrassingly parallel)
* Total throughput scales linearly with number of DP ranks

Formula: <code>total_gpus = dp_size * tp_size * pp_size</code>

=== External Launcher Mode ===
The <code>distributed_executor_backend="external_launcher"</code> setting:
* Tells vLLM that processes are already spawned by torchrun
* Each Python process creates exactly one worker
* Uses environment variables (RANK, WORLD_SIZE, etc.) to discover topology
* Enables integration with standard PyTorch distributed training tools

=== Load Balancing ===
The example implements simple modulo-based load balancing:
<syntaxhighlight lang="python">
dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

prompts = [
    f"{idx}.{prompt}"
    for idx, prompt in enumerate(prompts)
    if idx % dp_size == dp_rank
]
</syntaxhighlight>

Each rank processes prompts where <code>index % dp_size == rank</code>.

=== Multi-Dimensional Parallelism ===
Supports combining parallelism strategies:
* '''DP + TP''': Replicate large models that require tensor parallelism
* '''DP + PP''': Pipeline parallelism within groups, data parallelism across groups
* '''DP + TP + EP''': For mixture-of-experts models with tensor and expert parallelism

== Usage Examples ==

=== Basic Data Parallelism ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create LLM with external launcher
llm = LLM(
    model="microsoft/Phi-mini-MoE-instruct",
    data_parallel_size=2,
    tensor_parallel_size=1,
    distributed_executor_backend="external_launcher",
    seed=1,  # Ensures deterministic sampling across ranks
)

# Get this rank's data parallel ID
dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
dp_size = llm.llm_engine.vllm_config.parallel_config.data_parallel_size

# Partition prompts to this rank
my_prompts = [p for i, p in enumerate(prompts) if i % dp_size == dp_rank]

# Generate outputs
outputs = llm.generate(my_prompts, sampling_params)
for output in outputs:
    print(f"DP Rank {dp_rank}: {output.outputs[0].text}")
</syntaxhighlight>

=== Combining DP with Tensor Parallelism ===
<syntaxhighlight lang="bash">
# 8 GPUs: 2 DP ranks, each using TP=4
torchrun --nproc-per-node=8 \
    examples/offline_inference/torchrun_dp_example.py \
    --model meta-llama/Llama-3.1-70B \
    --dp-size=2 --tp-size=4 \
    --max-model-len=4096
</syntaxhighlight>

Configuration:
* Total processes: 8
* DP rank 0: GPUs 0-3 (TP group)
* DP rank 1: GPUs 4-7 (TP group)
* Each rank processes half the prompts
* 2x throughput compared to single TP=4 instance

=== Expert Parallelism with Data Parallelism ===
<syntaxhighlight lang="python">
# For MoE models like Mixtral or Phi-MoE
llm = LLM(
    model="mistralai/Mixtral-8x7B-v0.1",
    data_parallel_size=4,
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    max_model_len=32768,
    seed=42,
)

# 4 independent instances, each with TP=2 and EP enabled
# Total GPUs: 4 * 2 = 8
</syntaxhighlight>

=== Accessing PyTorch Process Groups ===
<syntaxhighlight lang="python">
from vllm.distributed.parallel_state import get_world_group

# CPU group (GLOO backend) for control messages
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)

if torch_rank == 0:
    # Only rank 0 saves results, logs, etc.
    save_results(outputs, "./results.json")

# Device group (NCCL backend) for GPU communication
device_group = get_world_group().device_group
# Use for custom all-reduce, broadcast, etc.
</syntaxhighlight>

=== Multi-Node Data Parallelism ===
<syntaxhighlight lang="bash">
# On master node (NODE_RANK=0)
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500

torchrun --nproc-per-node=8 \
    --nnodes=4 \
    --node-rank=0 \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    examples/offline_inference/torchrun_dp_example.py \
    --dp-size=32 --tp-size=1

# On worker nodes (NODE_RANK=1,2,3)
torchrun --nproc-per-node=8 \
    --nnodes=4 \
    --node-rank=$NODE_RANK \
    --master-addr=192.168.1.10 \
    --master-port=29500 \
    examples/offline_inference/torchrun_dp_example.py \
    --dp-size=32 --tp-size=1
</syntaxhighlight>

Result: 32 independent model instances across 4 nodes (8 GPUs each)

== Advanced Topics ==

=== Deterministic Sampling Across Ranks ===
The example sets an explicit seed:
<syntaxhighlight lang="python">
llm = LLM(
    model="...",
    seed=1,  # Critical for reproducibility
    distributed_executor_backend="external_launcher",
)
</syntaxhighlight>

Without a fixed seed, each DP rank would generate different outputs even for the same prompt, making comparison difficult.

=== Direct Model Access ===
For advanced use cases requiring direct model manipulation:
<syntaxhighlight lang="python">
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

# Access model parameters, layers, etc.
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
</syntaxhighlight>

=== Custom Load Distribution ===
Instead of simple modulo distribution, implement custom strategies:
<syntaxhighlight lang="python">
# Priority-based distribution
high_priority = [p for p in prompts if p.priority == "high"]
low_priority = [p for p in prompts if p.priority == "low"]

# Process high priority on all ranks first
if dp_rank < len(high_priority):
    my_prompts = [high_priority[dp_rank]]
else:
    idx = dp_rank - len(high_priority)
    if idx < len(low_priority):
        my_prompts = [low_priority[idx]]
</syntaxhighlight>

== Performance Considerations ==

=== Throughput Scaling ===
Expected throughput improvement:
* 2 DP ranks: ~1.9x throughput (vs. single instance)
* 4 DP ranks: ~3.8x throughput
* 8 DP ranks: ~7.6x throughput

Scaling efficiency: 95-98% (very high for embarrassingly parallel workloads)

=== Memory Requirements ===
Each DP rank loads a complete model copy:
* 7B model: ~14GB per rank
* 70B model with TP=4: ~20GB per rank (per GPU)
* Total memory: <code>model_size * dp_size</code>

=== Latency Characteristics ===
* Individual request latency unchanged (same as single instance)
* Aggregate throughput increases linearly
* Best for batch processing, not interactive serving

== Comparison with Other Patterns ==

{| class="wikitable"
|-
! Pattern !! Use Case !! Scaling !! Memory
|-
| Data Parallelism || High throughput batch inference || Linear || N × model size
|-
| Tensor Parallelism || Large models that don't fit on 1 GPU || Sub-linear || 1 × model size (sharded)
|-
| Pipeline Parallelism || Very large models with many layers || Sub-linear || 1 × model size (sharded)
|-
| DP + TP || Large models + high throughput || Near-linear || DP × (model size / TP)
|}

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_TorchrunTensorParallelInference]]
* [[related::Concept:vllm-project_vllm_Data_Parallelism]]
* [[related::Concept:vllm-project_vllm_Tensor_Parallelism]]
* [[related::Concept:vllm-project_vllm_Distributed_Inference]]
