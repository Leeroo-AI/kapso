{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Paper|S-LoRA: Serving Thousands of Concurrent LoRA Adapters|https://arxiv.org/abs/2311.03285]]
|-
! Domains
| [[domain::NLP]], [[domain::Scheduling]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

Pattern documentation for the internal scheduler logic that groups requests by LoRA adapter while respecting the max_loras constraint.

=== Description ===

The vLLM scheduler handles mixed-adapter batching by:
- Tracking which adapters are loaded in GPU memory
- Grouping requests to minimize adapter swapping
- Respecting the `max_loras` constraint per batch
- Balancing throughput with adapter switching overhead

This is internal implementation behavior—users don't directly interact with it but benefit from understanding how batching affects performance.

=== Usage ===

Understand scheduler behavior when:
- Optimizing multi-adapter serving throughput
- Tuning `max_loras` configuration
- Debugging adapter switching latency
- Designing request routing strategies

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project/vllm]
* '''File:''' vllm/core/scheduler.py
* '''Lines:''' L1-500

=== Pattern Specification ===
<syntaxhighlight lang="python">
# Conceptual scheduler behavior (not exact implementation)
class LoRABatchingScheduler:
    """
    Scheduler that groups requests by LoRA adapter.

    Key constraints:
    - max_loras: Maximum adapters in a single batch
    - Adapter loading overhead: Avoid frequent swapping
    - Memory pressure: May need to evict adapters

    Scheduling strategies:
    1. FCFS with adapter grouping: Process in order, group same adapter
    2. Adapter affinity: Prioritize requests matching loaded adapters
    3. Batch completion: Finish current batch before loading new adapters
    """

    def schedule_iteration(self) -> SchedulerOutput:
        """Form next batch respecting adapter constraints."""

        # 1. Get running sequences (priority)
        running = self.get_running_sequences()

        # 2. Get unique adapters in running batch
        running_adapters = {seq.lora_request.int_id
                          for seq in running if seq.lora_request}

        # 3. Admit waiting sequences if compatible
        waiting = self.get_waiting_sequences()
        for seq in waiting:
            adapter_id = seq.lora_request.int_id if seq.lora_request else 0

            # Check adapter constraint
            if adapter_id not in running_adapters:
                if len(running_adapters) >= self.max_loras:
                    continue  # Would exceed max_loras

            # Check memory constraint
            if not self.can_allocate(seq):
                continue

            # Admit to batch
            running.append(seq)
            running_adapters.add(adapter_id)

        return SchedulerOutput(running)
</syntaxhighlight>

== I/O Contract ==

=== Inputs (Internal) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| pending_requests || Queue || Requests waiting to be scheduled
|-
| running_requests || Set || Currently running sequences
|-
| max_loras || int || Maximum adapters per batch
|-
| loaded_adapters || Set || Adapters currently in GPU memory
|}

=== Outputs (Internal) ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| scheduled_batch || list || Requests to process this iteration
|-
| preempted || list || Requests moved back to waiting
|-
| adapter_loads || list || Adapters to load for this batch
|}

== Usage Examples ==

=== Understanding Batch Formation ===
<syntaxhighlight lang="python">
# Example: How scheduler forms batches
# Given max_loras=2 and these requests:
requests = [
    ("req1", "adapter_A"),
    ("req2", "adapter_B"),
    ("req3", "adapter_A"),  # Same as req1
    ("req4", "adapter_C"),  # Third adapter!
    ("req5", "adapter_B"),  # Same as req2
]

# Batch 1: req1, req2, req3, req5 (adapters A and B)
# Batch 2: req4 (adapter C, after A/B complete)

# req3 and req5 are grouped with matching adapters
# req4 waits because it would exceed max_loras=2
</syntaxhighlight>

=== Optimizing for Adapter Locality ===
<syntaxhighlight lang="python">
# Client-side request ordering can help scheduler
# Group requests by expected adapter before submitting

from collections import defaultdict

def batch_by_adapter(requests):
    """Group requests by adapter for better scheduling."""
    by_adapter = defaultdict(list)

    for prompt, adapter in requests:
        adapter_name = adapter.lora_name if adapter else "base"
        by_adapter[adapter_name].append((prompt, adapter))

    # Flatten, keeping adapter groups together
    ordered = []
    for prompts in by_adapter.values():
        ordered.extend(prompts)

    return ordered

# Submit in grouped order
ordered_requests = batch_by_adapter(requests)
outputs = llm.generate(
    [p for p, _ in ordered_requests],
    lora_request=[a for _, a in ordered_requests],
)
</syntaxhighlight>

=== Monitoring Adapter Switches ===
<syntaxhighlight lang="python">
# Log adapter usage patterns
import logging

logger = logging.getLogger("adapter_monitor")

class AdapterTracker:
    def __init__(self):
        self.last_adapter = None
        self.switches = 0

    def track(self, adapter_name):
        if adapter_name != self.last_adapter:
            if self.last_adapter is not None:
                self.switches += 1
                logger.info(f"Adapter switch: {self.last_adapter} -> {adapter_name}")
            self.last_adapter = adapter_name

    def report(self):
        logger.info(f"Total adapter switches: {self.switches}")
</syntaxhighlight>

=== Tuning max_loras ===
<syntaxhighlight lang="python">
# Trade-off analysis
# Higher max_loras:
#   + More adapter diversity per batch
#   + Better for many small adapters
#   - More memory for adapter weights
#   - Potential compute overhead

# Lower max_loras:
#   + Less memory per batch
#   + Simpler compute graph
#   - More frequent adapter switching
#   - Higher latency for mixed workloads

# Recommendation: Start with max_loras=4, tune based on workload
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,  # Good starting point
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:vllm-project_vllm_LoRA_Scheduling]]

=== Requires Environment ===
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
