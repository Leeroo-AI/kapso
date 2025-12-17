# Batch Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|vLLM: PagedAttention|https://arxiv.org/abs/2309.06180]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Inference]], [[domain::Generation]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for executing efficient batch text generation across multiple prompts using continuous batching and PagedAttention memory management.

=== Description ===

Batch Generation is the core inference operation in vLLM, leveraging two key innovations:

1. **Continuous Batching**: Instead of waiting for all sequences in a batch to complete, new requests can join an ongoing batch as soon as GPU resources become available. This maximizes throughput by avoiding idle GPU time.

2. **PagedAttention**: KV cache memory is managed in fixed-size blocks (like virtual memory pages), allowing efficient memory sharing across sequences and eliminating fragmentation.

The batch generation process:
* Accepts multiple prompts with varying lengths
* Dynamically schedules tokens across the batch
* Manages memory through block-level allocation
* Returns results as sequences complete

=== Usage ===

Execute batch generation when:
* Processing multiple prompts together for maximum throughput
* Running offline inference workloads
* Benchmarking model performance
* Processing datasets for evaluation or data generation

Batch generation automatically handles memory constraints, scheduling, and output ordering.

== Theoretical Basis ==

'''Continuous Batching Schedule:'''
<syntaxhighlight lang="python">
# Abstract continuous batching algorithm
def continuous_batch_step(active_requests, new_requests, memory_pool):
    # Finish completed sequences
    completed = [r for r in active_requests if r.is_done()]
    for r in completed:
        release_memory(r, memory_pool)
        yield r.output

    # Add new requests if memory available
    active = [r for r in active_requests if not r.is_done()]
    for new_req in new_requests:
        if can_allocate(new_req, memory_pool):
            allocate_memory(new_req, memory_pool)
            active.append(new_req)

    # Execute one forward pass for all active
    if active:
        run_forward_pass(active)
</syntaxhighlight>

'''PagedAttention Block Management:'''
* Block size: Typically 16 tokens
* Allocation: On-demand as sequences grow
* Sharing: Copy-on-write for beam search and prefix caching

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_generate]]
