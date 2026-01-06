{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Data Parallelism]], [[domain::Distributed Inference]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates native data parallelism in vLLM for processing disjoint datasets across multiple GPU replicas.

=== Description ===
The data parallel example shows how to scale inference throughput by distributing different prompts across multiple independent model replicas. Unlike tensor parallelism (which splits a single model across GPUs), data parallelism creates complete model copies on different GPU sets, each processing its own subset of the workload. The example supports both single-node and multi-node deployments, with each data parallel rank potentially using tensor parallelism internally. This approach maximizes throughput for large batch workloads.

=== Usage ===
Use data parallelism when you need to maximize throughput for large batch inference workloads, have sufficient GPU resources to deploy multiple model replicas, or want to process independent request streams in parallel. This is particularly effective for offline batch processing scenarios.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py examples/offline_inference/data_parallel.py]

=== CLI Usage ===

==== Single Node ====
<syntaxhighlight lang="bash">
python examples/offline_inference/data_parallel.py \
    --model="ibm-research/PowerMoE-3b" \
    --dp-size=2 \
    --tp-size=2
</syntaxhighlight>

==== Multi-Node ====
<syntaxhighlight lang="bash">
# Node 0 (master at 10.99.48.128)
python examples/offline_inference/data_parallel.py \
    --model="ibm-research/PowerMoE-3b" \
    --dp-size=2 \
    --tp-size=2 \
    --node-size=2 \
    --node-rank=0 \
    --master-addr=10.99.48.128 \
    --master-port=13345

# Node 1
python examples/offline_inference/data_parallel.py \
    --model="ibm-research/PowerMoE-3b" \
    --dp-size=2 \
    --tp-size=2 \
    --node-size=2 \
    --node-rank=1 \
    --master-addr=10.99.48.128 \
    --master-port=13345
</syntaxhighlight>

== Key Concepts ==

=== Data Parallel Environment Variables ===
Each data parallel rank is configured via environment variables:
* '''VLLM_DP_RANK''': Global rank of this DP instance
* '''VLLM_DP_RANK_LOCAL''': Local rank within the node
* '''VLLM_DP_SIZE''': Total number of DP replicas
* '''VLLM_DP_MASTER_IP''': IP address of coordination master
* '''VLLM_DP_MASTER_PORT''': Port for coordination

=== Workload Distribution ===
The example demonstrates proper data partitioning:
* Prompts are evenly distributed across DP ranks
* Each rank processes a disjoint subset of prompts
* Remainder prompts are distributed to avoid imbalance
* Placeholder prompts handle edge cases where a rank has no work

=== Combining DP and TP ===
Data parallelism can be combined with tensor parallelism:
* '''DP Size''': Number of independent model replicas
* '''TP Size''': GPUs per model replica
* '''Total GPUs''': dp_size × tp_size
* Example: dp_size=2, tp_size=2 requires 4 GPUs total

== Usage Examples ==

<syntaxhighlight lang="python">
import os
from vllm import LLM, SamplingParams

# Configure data parallel environment
os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
os.environ["VLLM_DP_SIZE"] = str(dp_size)
os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

# Partition prompts across DP ranks
prompts = ["prompt1", "prompt2", ...] * 100
floor = len(prompts) // dp_size
remainder = len(prompts) % dp_size

def start(rank):
    return rank * floor + min(rank, remainder)

prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]

# Each DP rank can have different sampling params
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=[16, 20][global_dp_rank % 2]
)

# Create LLM with tensor parallelism per replica
llm = LLM(
    model="ibm-research/PowerMoE-3b",
    tensor_parallel_size=tp_size,
    max_num_seqs=64,
)

outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

=== Multi-Process Launch Pattern ===
<syntaxhighlight lang="python">
from multiprocessing import Process

procs = []
for local_dp_rank, global_dp_rank in enumerate(
    range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
):
    proc = Process(
        target=main,
        args=(model, dp_size, local_dp_rank, global_dp_rank, ...)
    )
    proc.start()
    procs.append(proc)

# Wait for all processes with timeout
for proc in procs:
    proc.join(timeout=300)
    if proc.exitcode is None:
        proc.kill()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
