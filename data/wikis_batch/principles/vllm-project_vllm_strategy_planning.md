= Parallelism Strategy Planning =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vLLM Documentation, vllm/config/parallel.py, examples/offline_inference/data_parallel.py
|-
| Domains || Distributed Systems, Model Parallelism, Resource Management
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The Parallelism Strategy Planning principle defines the decision-making process for selecting and configuring parallelism strategies in vLLM. This principle guides users in determining the optimal combination of Tensor Parallelism (TP), Data Parallelism (DP), Pipeline Parallelism (PP), and Expert Parallelism (EP) based on model architecture, available hardware, and workload characteristics.

== Description ==
vLLM supports multiple parallelism strategies that can be combined to scale inference across multiple GPUs and nodes:

=== Parallelism Types ===
* '''Tensor Parallelism (TP)''': Splits model weights across GPUs within a single node. Each GPU holds a portion of the model and performs computation on the same input data. Best for large models that don't fit in single GPU memory.
* '''Data Parallelism (DP)''': Creates multiple independent model replicas, each processing different subsets of input data. Ideal for high-throughput scenarios with many concurrent requests.
* '''Pipeline Parallelism (PP)''': Distributes model layers across GPUs, processing different stages of the computation pipeline. Useful for very large models with many layers.
* '''Expert Parallelism (EP)''': Specifically for Mixture-of-Experts (MoE) models, distributes experts across GPUs to enable efficient sparse computation.

=== Strategy Selection Criteria ===
The optimal parallelism strategy depends on several factors:

# '''Model Size''': Large models require TP or PP to fit in memory
# '''Model Architecture''': MoE models benefit from EP; standard transformers use TP/DP/PP
# '''Hardware Topology''': Single node vs. multi-node, GPU memory capacity, interconnect bandwidth
# '''Workload Characteristics''': Batch size, sequence length, throughput requirements
# '''Scalability Goals''': Maximize throughput vs. minimize latency

=== Planning Guidelines ===
* Start with TP to split large models across GPUs within a node
* Add DP to scale throughput by creating multiple replicas
* Use EP for MoE models to parallelize expert computation
* Ensure total GPUs = TP × DP × PP × (EP if applicable)
* Consider communication overhead: TP requires high-bandwidth GPU interconnect

== Configuration Parameters ==
Key parameters for strategy planning:

* <code>tensor_parallel_size</code>: Number of GPUs for tensor parallelism
* <code>data_parallel_size</code>: Number of data parallel replicas
* <code>pipeline_parallel_size</code>: Number of pipeline stages
* <code>enable_expert_parallel</code>: Enable expert parallelism for MoE models
* <code>expert_placement_strategy</code>: Strategy for distributing experts ("linear" or "round_robin")

== Usage Patterns ==

=== Single-Node Scaling ===
For a model that fits in 2 GPUs with TP, scale to 8 GPUs using DP:
<syntaxhighlight lang="python">
# 8 GPUs total: 2 for TP, 4 for DP
# This creates 4 model replicas, each using 2 GPUs
tensor_parallel_size = 2
data_parallel_size = 4
</syntaxhighlight>

=== Multi-Node Scaling ===
For a large model across 2 nodes with 8 GPUs each:
<syntaxhighlight lang="python">
# 16 GPUs total across 2 nodes
# TP=4 (model split across 4 GPUs)
# DP=4 (4 replicas, 2 per node)
tensor_parallel_size = 4
data_parallel_size = 4
node_size = 2
</syntaxhighlight>

=== MoE Model Strategy ===
For Mixture-of-Experts models:
<syntaxhighlight lang="python">
# Enable expert parallelism to distribute experts
# TP=2 for base model layers, EP for expert layers
tensor_parallel_size = 2
data_parallel_size = 2
enable_expert_parallel = True
expert_placement_strategy = "round_robin"
</syntaxhighlight>

== Design Rationale ==
The strategy planning principle addresses key challenges in distributed inference:

* '''Memory Efficiency''': TP and PP enable running models larger than single GPU memory
* '''Throughput Scaling''': DP scales throughput linearly with number of replicas
* '''Hardware Utilization''': Optimal strategy maximizes GPU compute while minimizing communication overhead
* '''Flexibility''': Combining strategies allows adaptation to different hardware topologies

The principle emphasizes starting with simple strategies (e.g., TP only) and progressively adding complexity (e.g., TP + DP) as scaling requirements grow.

== Related Pages ==
* [[implemented_by::Implementation:vllm-project_vllm_ParallelConfig]] - Environment variables that configure the selected strategy
* [[related_to::ParallelConfig]] - Configuration class that stores parallelism settings
* [[related_to::Distributed_Data_Parallel_Inference]] - Workflow that implements the strategy

== See Also ==
* vLLM Distributed Inference Documentation
* vllm/config/parallel.py - ParallelConfig implementation
* examples/offline_inference/data_parallel.py - Reference implementation
