{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|S-LoRA: Serving Thousands of Concurrent LoRA Adapters|https://arxiv.org/abs/2311.03285]]
|-
! Domains
| [[domain::NLP]], [[domain::LoRA]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-01-15 10:00 GMT]]
|}

== Overview ==

The process of configuring an inference engine to support dynamic loading and serving of multiple LoRA (Low-Rank Adaptation) adapters on a single base model.

=== Description ===

LoRA Engine Configuration prepares the inference engine to handle multiple fine-tuned model variants efficiently. Key considerations include:

1. **Memory Allocation:** Reserving space for LoRA weight matrices
2. **Rank Support:** Maximum rank determines memory per adapter
3. **Concurrency:** How many adapters can be in a single batch
4. **Caching Strategy:** CPU/GPU adapter swapping policies
5. **Vocabulary Extension:** Supporting adapter-specific tokens

This configuration trades memory for flexibility, enabling multi-tenant serving without loading separate model copies.

=== Usage ===

Configure LoRA engine support when:
- Serving multiple fine-tuned variants of a base model
- Building multi-tenant LLM platforms
- A/B testing different fine-tuned versions
- Reducing GPU memory usage vs. separate model copies
- Supporting user-specific personalized models

== Theoretical Basis ==

'''LoRA Memory Model:'''

LoRA adds low-rank decomposition matrices to attention weights:

<math>
W' = W + BA
</math>

Where <math>B \in \mathbb{R}^{d \times r}</math> and <math>A \in \mathbb{R}^{r \times k}</math> with rank <math>r \ll \min(d, k)</math>.

Memory per adapter:
<math>
Memory_{LoRA} = 2 \times r \times (d_{model} + d_{ff}) \times n_{layers} \times dtype\_size
</math>

For Llama-7B with r=16 and float16:
<math>
Memory \approx 2 \times 16 \times (4096 + 11008) \times 32 \times 2 \approx 31MB
</math>

'''Engine Configuration Trade-offs:'''

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| max_loras | More concurrent adapters | Less memory |
| max_lora_rank | Support higher-rank adapters | More memory per adapter |
| max_cpu_loras | Faster adapter swapping | More CPU RAM |
| lora_extra_vocab_size | More adapter-specific tokens | More vocab memory |

'''Adapter Management:'''

<syntaxhighlight lang="python">
# Conceptual adapter lifecycle
class LoRAManager:
    def __init__(self, max_loras, max_cpu_loras):
        self.gpu_cache = LRUCache(max_loras)
        self.cpu_cache = LRUCache(max_cpu_loras)

    def get_adapter(self, lora_request):
        # Check GPU cache
        if lora_request.name in self.gpu_cache:
            return self.gpu_cache[lora_request.name]

        # Check CPU cache
        if lora_request.name in self.cpu_cache:
            # Move to GPU (may evict another)
            adapter = self.cpu_cache[lora_request.name]
            self._load_to_gpu(adapter)
            return adapter

        # Load from disk
        adapter = load_adapter(lora_request.path)
        self._load_to_gpu(adapter)
        return adapter
</syntaxhighlight>

'''Sharded LoRA for Tensor Parallelism:'''

When using multiple GPUs, LoRA weights can be sharded:

<syntaxhighlight lang="python">
# Sharding strategy (conceptual)
# With TP=4 for attention with 32 heads:
# GPU 0: heads 0-7, LoRA B[:, :r], A[:r, :k/4]
# GPU 1: heads 8-15, LoRA B[:, :r], A[:r, k/4:k/2]
# etc.
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_EngineArgs_lora]]

