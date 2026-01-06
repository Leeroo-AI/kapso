{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|S-LoRA: Serving Thousands of Concurrent LoRA Adapters|https://arxiv.org/abs/2311.03285]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Request_Processing]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of submitting an inference request with an associated LoRA adapter specification to the engine for processing.

=== Description ===

LoRA Request Submission is the point where a prompt and its adapter choice enter the inference pipeline. Key aspects:

1. **Adapter Binding:** Associating a request with its LoRA adapter
2. **Batch Formation:** Grouping requests, potentially with different adapters
3. **Adapter Loading:** Triggering on-demand loading if not cached
4. **Queue Management:** Prioritizing and ordering mixed-adapter batches

This step bridges the registration of adapters (what adapters exist) with their actual use (which adapter for this specific request).

=== Usage ===

Submit LoRA-enabled requests when:
- Processing user requests with personalized adapters
- Running A/B tests with different model versions
- Implementing task-specific routing logic
- Batching diverse workloads on shared infrastructure

== Theoretical Basis ==

'''Request Association:'''

Each request carries its adapter reference:

<syntaxhighlight lang="python">
# Conceptual request structure
request = {
    "request_id": "req-123",
    "prompt": "Convert to SQL: ...",
    "sampling_params": SamplingParams(...),
    "lora_request": LoRARequest("sql", 1, "path/to/adapter"),
}
</syntaxhighlight>

'''Adapter Loading on Demand:'''

<syntaxhighlight lang="python">
# Conceptual loading flow
async def process_request(request):
    adapter = request.lora_request

    if adapter:
        # Check if adapter is loaded
        if not is_loaded(adapter):
            # Load to GPU (may evict others)
            await load_adapter(adapter)

        # Apply adapter weights
        with adapter_context(adapter):
            output = await generate(request)
    else:
        # Use base model
        output = await generate(request)

    return output
</syntaxhighlight>

'''Batch Formation with Mixed Adapters:'''

The scheduler groups requests by adapter compatibility:

<syntaxhighlight lang="python">
# Conceptual batch formation
def form_batch(pending_requests, max_loras):
    batch = []
    adapters_in_batch = set()

    for request in pending_requests:
        adapter_id = request.lora_request.int_id if request.lora_request else 0

        # Can we add this adapter to the batch?
        if adapter_id not in adapters_in_batch:
            if len(adapters_in_batch) >= max_loras:
                break  # Batch full of adapters

        batch.append(request)
        adapters_in_batch.add(adapter_id)

    return batch
</syntaxhighlight>

'''Per-Prompt vs Single Adapter:'''

Two modes of adapter specification:

<syntaxhighlight lang="python">
# Mode 1: Single adapter for all prompts
llm.generate(
    prompts=["prompt1", "prompt2", "prompt3"],
    lora_request=single_adapter,  # Applied to all
)

# Mode 2: Per-prompt adapters
llm.generate(
    prompts=["prompt1", "prompt2", "prompt3"],
    lora_request=[adapter1, adapter2, adapter3],  # One per prompt
)

# Mode 3: Mixed (some with adapter, some without)
llm.generate(
    prompts=["prompt1", "prompt2", "prompt3"],
    lora_request=[adapter1, None, adapter2],  # None = base model
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLMEngine_add_request_lora]]
