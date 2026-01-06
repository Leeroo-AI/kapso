{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|S-LoRA: Serving Thousands of Concurrent LoRA Adapters|https://arxiv.org/abs/2311.03285]]
* [[source::Paper|vLLM: Easy, Fast, and Cheap LLM Serving|https://arxiv.org/abs/2309.06180]]
|-
! Domains
| [[domain::NLP]], [[domain::Scheduling]], [[domain::Systems]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The scheduling strategy for batching inference requests that use different LoRA adapters while minimizing adapter switching overhead.

=== Description ===

LoRA Scheduling addresses the challenge of efficiently serving requests that use different adapters. Key challenges include:

1. **Adapter Diversity:** Requests may use different adapters
2. **Memory Constraints:** Only N adapters fit in GPU memory simultaneously
3. **Switching Overhead:** Loading new adapters has cost
4. **Fairness:** All requests should be served, not just popular adapters
5. **Throughput:** Maximize GPU utilization despite constraints

The scheduler balances these concerns to achieve high throughput while maintaining reasonable latency.

=== Usage ===

Understand LoRA scheduling when:
- Designing multi-tenant adapter serving systems
- Tuning max_loras for workload characteristics
- Optimizing request routing and batching
- Debugging latency issues in mixed-adapter workloads

== Theoretical Basis ==

'''Problem Formulation:'''

Given:
- Requests <math>R = \{r_1, r_2, ..., r_n\}</math>
- Adapters <math>A = \{a_1, a_2, ..., a_m\}</math>
- Constraint <math>max\_loras = k</math>

Minimize total completion time while ensuring:
- No batch has more than <math>k</math> distinct adapters
- All requests eventually complete
- Memory constraints are satisfied

'''Scheduling Strategies:'''

<syntaxhighlight lang="python">
# Strategy 1: FCFS with adapter grouping
def fcfs_grouped(requests, max_loras):
    """Process in order, skip incompatible adapters."""
    batch = []
    adapters = set()

    for req in requests:
        adapter = req.adapter_id
        if adapter in adapters or len(adapters) < max_loras:
            batch.append(req)
            adapters.add(adapter)

    return batch

# Strategy 2: Adapter affinity
def affinity_scheduling(requests, loaded_adapters, max_loras):
    """Prioritize requests matching loaded adapters."""
    # First: requests using already-loaded adapters
    priority = [r for r in requests if r.adapter_id in loaded_adapters]
    # Then: other requests (may need adapter loading)
    rest = [r for r in requests if r.adapter_id not in loaded_adapters]
    return priority + rest

# Strategy 3: Batch completion
def batch_completion(requests, running_adapters):
    """Finish current adapter batch before switching."""
    # Only admit requests matching current adapters
    return [r for r in requests if r.adapter_id in running_adapters]
</syntaxhighlight>

'''Adapter Loading Cost:'''

Loading an adapter involves:
1. CPU → GPU memory transfer
2. Potential eviction of another adapter
3. Cache invalidation

<math>
Cost_{switch} \approx T_{transfer} + T_{eviction} \approx 10-100ms
</math>

This makes batching same-adapter requests valuable.

'''Memory Model:'''

<math>
GPU_{adapters} = \sum_{i=1}^{max\_loras} Memory(adapter_i)
</math>

Where:
<math>
Memory(adapter) = 2 \times rank \times hidden \times layers \times dtype\_bytes
</math>

'''Scheduling Algorithm (Conceptual):'''

<syntaxhighlight lang="python">
def schedule_lora_batch(waiting, running, max_loras, memory_budget):
    """
    Form optimal batch respecting LoRA constraints.

    Returns:
        batch: Requests to process
        to_load: Adapters to load
        to_evict: Adapters to evict
    """
    batch = list(running)  # Keep running sequences
    adapters_in_batch = {r.adapter_id for r in running}

    # Score waiting requests
    for req in sorted(waiting, key=lambda r: score(r, adapters_in_batch)):
        # Can we add this request?
        new_adapter = req.adapter_id not in adapters_in_batch

        if new_adapter and len(adapters_in_batch) >= max_loras:
            continue  # Would exceed adapter limit

        if not fits_in_memory(req, memory_budget):
            continue  # Would exceed memory

        batch.append(req)
        adapters_in_batch.add(req.adapter_id)

    # Determine adapter operations
    to_load = adapters_in_batch - loaded_adapters
    to_evict = select_evictions(to_load, loaded_adapters, max_cpu_loras)

    return batch, to_load, to_evict

def score(request, current_adapters):
    """Score request for scheduling priority."""
    # Prefer requests matching current adapters (no switch cost)
    if request.adapter_id in current_adapters:
        return 0
    # Otherwise, FIFO by arrival time
    return request.arrival_time
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_Scheduler_lora_batching]]

