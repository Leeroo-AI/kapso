{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Distributed_Computing]], [[domain::Inference]], [[domain::Parallelism]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for scaling LLM inference across multiple GPUs and nodes using data parallelism, tensor parallelism, and expert parallelism strategies.

=== Description ===

This workflow demonstrates vLLM's distributed inference capabilities for scaling throughput across multiple GPUs and compute nodes. It supports multiple parallelism strategies that can be combined for optimal hardware utilization.

Key parallelism modes:
* **Tensor Parallelism (TP)**: Split model layers across GPUs for large models
* **Data Parallelism (DP)**: Replicate model across GPU groups for throughput
* **Expert Parallelism (EP)**: Distribute MoE experts across GPUs
* **Pipeline Parallelism**: Stage-based distribution (via external tools)

=== Usage ===

Execute this workflow when you need to:
* Run models too large for single GPU memory
* Scale throughput across multiple GPUs
* Deploy across multiple compute nodes
* Handle high-concurrency inference workloads
* Serve Mixture-of-Experts models efficiently

Ideal for production deployments requiring high throughput or large model support.

== Execution Steps ==

=== Step 1: Plan Parallelism Strategy ===
[[step::Principle:vllm-project_vllm_strategy_planning]]

Determine the optimal parallelism configuration based on model size, available hardware, and throughput requirements. Consider GPU memory, interconnect bandwidth, and workload characteristics.

'''Strategy guidelines:'''
* **Single GPU fits model**: No parallelism needed
* **Model too large for one GPU**: Use tensor parallelism (TP)
* **Need higher throughput**: Add data parallelism (DP)
* **MoE model**: Enable expert parallelism (EP)
* Total GPUs = DP_size × TP_size

=== Step 2: Configure Environment Variables ===
[[step::Principle:vllm-project_vllm_dp_env_vars]]

Set up environment variables for distributed coordination. This includes rank information, master node address, and communication ports.

'''Environment variables:'''
* `VLLM_DP_RANK`: Data parallel rank (0-indexed)
* `VLLM_DP_SIZE`: Total data parallel replicas
* `VLLM_DP_MASTER_IP`: Coordinator node IP address
* `VLLM_DP_MASTER_PORT`: Coordination port

=== Step 3: Initialize Distributed Engine ===
[[step::Principle:vllm-project_vllm_LLM_distributed]]

Create LLM instances across distributed workers. Each data parallel rank runs an independent LLM instance, while tensor parallel ranks coordinate within each replica.

'''Initialization per rank:'''
1. Set rank-specific environment
2. Create LLM with `tensor_parallel_size` matching TP config
3. Engine auto-coordinates with other TP ranks
4. Each DP rank processes independent prompts

=== Step 4: Partition Data Across DP Ranks ===
[[step::Principle:vllm-project_vllm_prompt_partitioning]]

Distribute input prompts across data parallel ranks. Each rank processes a disjoint subset of the total workload, enabling linear throughput scaling.

'''Partitioning approach:'''
* Divide prompts evenly across DP ranks
* Handle remainder distribution
* Ensure no rank has empty workload
* Each rank can have different sampling params

=== Step 5: Execute Parallel Inference ===
[[step::Principle:vllm-project_vllm_LLM_generate_dp]]

Launch generation across all distributed workers. Each DP rank processes its partition independently while TP coordination happens within each replica.

'''Execution flow:'''
1. Each DP rank receives its prompt subset
2. Tensor parallel workers coordinate forward passes
3. Continuous batching within each replica
4. Results collected per-rank

=== Step 6: Aggregate Distributed Results ===
[[step::Principle:vllm-project_vllm_result_aggregation]]

Collect and merge outputs from all data parallel ranks. Results maintain association with original prompts for correct ordering.

'''Aggregation process:'''
* Each rank outputs for its partition
* Master process collects all outputs
* Reorder results to match original input order
* Handle any rank failures gracefully

== Execution Diagram ==
{{#mermaid:graph TD
    A[Plan Parallelism Strategy] --> B[Configure Environment Variables]
    B --> C[Initialize Distributed Engine]
    C --> D[Partition Data Across DP Ranks]
    D --> E[Execute Parallel Inference]
    E --> F[Aggregate Distributed Results]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_strategy_planning]]
* [[step::Principle:vllm-project_vllm_dp_env_vars]]
* [[step::Principle:vllm-project_vllm_LLM_distributed]]
* [[step::Principle:vllm-project_vllm_prompt_partitioning]]
* [[step::Principle:vllm-project_vllm_LLM_generate_dp]]
* [[step::Principle:vllm-project_vllm_result_aggregation]]
