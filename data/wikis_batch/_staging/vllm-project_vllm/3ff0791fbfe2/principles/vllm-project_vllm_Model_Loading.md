# Model Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|vLLM: PagedAttention|https://arxiv.org/abs/2309.06180]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Inference]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:30 GMT]]
|}

== Overview ==

Principle for instantiating and initializing the LLM inference engine with model weights, tokenizer, and KV cache memory allocation.

=== Description ===

Model Loading encompasses the complete process of preparing an LLM for inference:

1. **Model discovery**: Resolving model path from HuggingFace Hub or local filesystem
2. **Weight loading**: Loading and optionally quantizing model parameters
3. **Tokenizer initialization**: Loading tokenizer with proper configuration
4. **Memory allocation**: Reserving GPU memory for KV cache using PagedAttention
5. **Engine initialization**: Creating the inference engine with configured workers

vLLM's model loading is optimized for production deployment with features like:
* Automatic dtype selection based on model config
* Quantization integration (AWQ, GPTQ, FP8)
* Distributed loading for tensor parallelism
* Memory-efficient KV cache pre-allocation

=== Usage ===

Load models when:
* Initializing an offline inference pipeline
* Starting a new inference server
* Switching between different models or configurations
* Loading quantized models for memory efficiency

Model loading is typically a one-time operation at startup, as the loaded engine can serve many inference requests.

== Theoretical Basis ==

'''PagedAttention Memory Management:'''

vLLM's key innovation is PagedAttention, which manages KV cache memory like virtual memory:

<syntaxhighlight lang="python">
# Abstract algorithm for KV cache allocation
def allocate_kv_cache(gpu_memory, memory_utilization, block_size):
    available = gpu_memory * memory_utilization
    model_memory = estimate_model_memory()
    kv_memory = available - model_memory
    num_blocks = kv_memory // (block_size * kv_per_block)
    return allocate_blocks(num_blocks)
</syntaxhighlight>

'''Weight Distribution for Tensor Parallelism:'''
* Column-parallel: Weight matrices split along output dimension
* Row-parallel: Weight matrices split along input dimension
* Proper communication patterns ensure correct gradients/outputs

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:vllm-project_vllm_LLM_init]]
