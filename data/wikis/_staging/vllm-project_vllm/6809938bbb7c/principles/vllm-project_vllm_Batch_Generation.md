{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|vLLM: Easy, Fast, and Cheap LLM Serving|https://arxiv.org/abs/2309.06180]]
* [[source::Paper|Efficient Memory Management for LLM Serving with PagedAttention|https://arxiv.org/abs/2309.06180]]
|-
! Domains
| [[domain::NLP]], [[domain::Inference]], [[domain::Systems]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The technique of processing multiple prompts simultaneously through a language model to maximize GPU utilization and throughput.

=== Description ===

Batch Generation is the core inference strategy in vLLM that enables high-throughput text generation. Instead of processing prompts one at a time, multiple prompts are grouped into batches and processed in parallel through the model.

Key innovations in vLLM's batch generation:
- **PagedAttention:** Memory-efficient KV cache management via paging
- **Continuous Batching:** Dynamic addition/removal of sequences mid-batch
- **Iteration-Level Scheduling:** Fine-grained control over sequence progress
- **Preemption:** Handle memory pressure by swapping sequences to CPU

These techniques together enable vLLM to achieve 24x higher throughput than naive batching approaches.

=== Usage ===

Apply batch generation when:
- Processing large numbers of prompts offline
- Running model evaluation benchmarks
- Building data processing pipelines with LLMs
- Maximizing GPU utilization for cost efficiency

Considerations:
- Larger batches improve throughput but increase latency per request
- Memory limits constrain maximum batch size
- Prompts with similar lengths batch more efficiently

== Theoretical Basis ==

'''PagedAttention Memory Model:'''

Traditional attention allocates contiguous memory for max sequence length. PagedAttention uses paging:

<math>
Memory_{traditional} = B \times L_{max} \times D_{head} \times N_{heads}
</math>

<math>
Memory_{paged} = \sum_{i=1}^{B} \lceil L_i / block\_size \rceil \times block\_size \times D_{head} \times N_{heads}
</math>

Where <math>B</math> is batch size, <math>L</math> is sequence length, and blocks are allocated on-demand.

'''Continuous Batching:'''

Unlike static batching that waits for all sequences to complete:

<syntaxhighlight lang="python">
# Static batching (inefficient)
def static_batch(prompts):
    outputs = model.forward(prompts)
    wait_until_all_complete(outputs)
    return outputs

# Continuous batching (vLLM approach)
def continuous_batch(prompts):
    running = set(prompts)
    completed = []
    while running:
        # Process one iteration
        outputs = model.forward_step(running)

        # Immediately return completed sequences
        for seq in outputs:
            if seq.is_finished():
                completed.append(seq)
                running.remove(seq)

        # Add new sequences if capacity available
        if has_capacity() and pending_queue:
            running.add(pending_queue.pop())

    return completed
</syntaxhighlight>

'''Throughput Analysis:'''

Throughput depends on:
- GPU compute (forward pass time)
- Memory bandwidth (KV cache access)
- Batch size (GPU utilization)

<math>
Throughput = \frac{B \times T_{output}}{T_{prefill} + T_{decode} \times T_{output}}
</math>

Where <math>T_{prefill}</math> is prompt processing time, <math>T_{decode}</math> is per-token decode time.

'''Scheduling Strategy:'''

<syntaxhighlight lang="python">
# Iteration-level scheduling (conceptual)
def schedule_iteration():
    # 1. Check memory budget
    available_blocks = total_blocks - used_blocks

    # 2. Schedule running sequences (priority)
    batch = []
    for seq in running_sequences:
        if seq.needs_block():
            if available_blocks > 0:
                allocate_block(seq)
                available_blocks -= 1
            else:
                preempt(seq)  # Swap to CPU
        batch.append(seq)

    # 3. Admit waiting sequences if capacity
    for seq in waiting_sequences:
        blocks_needed = estimate_blocks(seq)
        if blocks_needed <= available_blocks:
            batch.append(seq)
            available_blocks -= blocks_needed

    return batch
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_generate]]

