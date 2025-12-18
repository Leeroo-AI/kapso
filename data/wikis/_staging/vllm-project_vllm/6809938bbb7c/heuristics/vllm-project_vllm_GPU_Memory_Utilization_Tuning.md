# GPU Memory Utilization Tuning

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Memory Management|https://docs.vllm.ai/en/latest/serving/deployment.html]]
|-
! Domains
| [[domain::Optimization]], [[domain::GPU_Computing]], [[domain::Memory_Management]]
|-
! Last Updated
| [[last_updated::2025-01-15 14:00 GMT]]
|}

== Overview ==

Memory optimization technique for balancing KV cache size against model stability using the `gpu_memory_utilization` parameter.

=== Description ===

The `gpu_memory_utilization` parameter controls the fraction of GPU memory (between 0 and 1) that vLLM reserves for model weights, activations, and KV cache. The default value of 0.9 (90%) is optimized for maximum throughput on dedicated inference servers. However, this aggressive default can cause out-of-memory (OOM) errors in constrained environments or when running alongside other GPU processes.

=== Usage ===

Use this heuristic when you need to:
- **Prevent OOM errors** during model loading or inference
- **Share GPU** with other processes (e.g., training jobs, monitoring)
- **Run larger models** on limited VRAM by reducing cache allocation
- **Maximize throughput** by tuning cache size for your workload

== The Insight (Rule of Thumb) ==

* **Action:** Set `gpu_memory_utilization` parameter in `LLM()` constructor
* **Default Value:** 0.9 (90% of GPU memory)
* **Recommended Range:** 0.7 - 0.95 depending on use case
* **Trade-off:** Lower values reduce OOM risk but decrease maximum batch size and throughput

{| class="wikitable"
! Scenario !! Recommended Value !! Rationale
|-
| Dedicated inference server || 0.9 - 0.95 || Maximum throughput, full GPU utilization
|-
| Shared GPU environment || 0.5 - 0.7 || Leave headroom for other processes
|-
| OOM during loading || 0.7 - 0.8 || Reduce peak memory during initialization
|-
| Large context windows || 0.85 - 0.9 || More KV cache for longer sequences
|-
| Small model, high concurrency || 0.9 || Maximize KV cache for more concurrent requests
|-
| Running with monitoring tools || 0.8 - 0.85 || Account for profiler overhead
|}

== Reasoning ==

vLLM's memory is split between:
1. **Model weights** (fixed, depends on model size and dtype)
2. **Activations** (temporary, depends on batch size)
3. **KV cache** (the remaining memory after weights and activations)

The `gpu_memory_utilization` parameter determines how much total GPU memory vLLM can use. The KV cache is what enables high-throughput continuous batching - larger cache means more concurrent sequences can be processed.

'''Why 0.9 is the default:'''
- Leaves 10% buffer for CUDA context, memory fragmentation, and temporary allocations
- Maximizes KV cache capacity for throughput
- Works well on dedicated inference hardware

'''When to reduce:'''
- OOM errors during model loading (reduce to 0.7-0.8)
- OOM during inference with large batches (reduce by 0.05-0.1 increments)
- Running multiple processes on same GPU

== Code Evidence ==

Default value from `vllm/entrypoints/llm.py:208`:
<syntaxhighlight lang="python">
def __init__(
    self,
    model: str,
    ...
    gpu_memory_utilization: float = 0.9,  # Default 90%
    ...
)
</syntaxhighlight>

Documentation from `vllm/entrypoints/llm.py:135-139`:
<syntaxhighlight lang="python">
gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
    reserve for the model weights, activations, and KV cache. Higher
    values will increase the KV cache size and thus improve the model's
    throughput. However, if the value is too high, it may cause out-of-
    memory (OOM) errors.
</syntaxhighlight>

Alternative fine-grained control from `vllm/entrypoints/llm.py:140-147`:
<syntaxhighlight lang="python">
kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,
    this is set to None and vllm can automatically infer the kv cache
    size based on gpu_memory_utilization. However, users may want to
    manually specify the kv cache memory size. kv_cache_memory_bytes
    allows more fine-grain control of how much memory gets used when
    compared with using gpu_memory_utilization. Note that
    kv_cache_memory_bytes (when not-None) ignores
    gpu_memory_utilization
</syntaxhighlight>

== Usage Examples ==

=== Basic OOM Fix ===
<syntaxhighlight lang="python">
from vllm import LLM

# If you get OOM errors with default settings:
llm = LLM(
    model="meta-llama/Llama-3.2-7B",
    gpu_memory_utilization=0.8,  # Reduce from 0.9 to 0.8
)
</syntaxhighlight>

=== Shared GPU Environment ===
<syntaxhighlight lang="python">
# When sharing GPU with other processes
llm = LLM(
    model="meta-llama/Llama-3.2-7B",
    gpu_memory_utilization=0.5,  # Use only 50% of GPU
)
</syntaxhighlight>

=== Fine-Grained Control ===
<syntaxhighlight lang="python">
# Specify exact KV cache size instead
llm = LLM(
    model="meta-llama/Llama-3.2-7B",
    kv_cache_memory_bytes=8 * 1024 * 1024 * 1024,  # 8GB for KV cache
)
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_LLM_init]]
* [[uses_heuristic::Principle:vllm-project_vllm_LLM_Class_Initialization]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Basic_Offline_Inference]]
