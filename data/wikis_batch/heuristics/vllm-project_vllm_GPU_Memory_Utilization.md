# Heuristic: vllm-project_vllm_GPU_Memory_Utilization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Docs|https://docs.vllm.ai/en/latest/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Optimization]], [[domain::Infrastructure]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Memory optimization technique to tune `gpu_memory_utilization` parameter for optimal KV cache sizing and avoiding OOM errors.

=== Description ===

The `gpu_memory_utilization` parameter controls how much GPU memory vLLM reserves for the KV cache. Setting it too high can cause OOM errors, while setting it too low wastes GPU memory and reduces throughput. The default is 0.9 (90%), but optimal values depend on model size, GPU VRAM, and concurrent request count.

=== Usage ===

Use this heuristic when:
- Encountering **CUDA out of memory** errors during inference
- Running **large models** (70B+) on limited VRAM
- Serving **many concurrent requests** and need to maximize throughput
- Running **quantized models** that use less memory

== The Insight (Rule of Thumb) ==

* **Action:** Set `gpu_memory_utilization` in `LLM()` or `EngineArgs`
* **Default Value:** 0.9 (90%)
* **Conservative Value:** 0.85 for stability with variable workloads
* **Aggressive Value:** 0.95 for maximum throughput on dedicated inference servers
* **OOM Recovery:** If OOM occurs, reduce by 0.05 increments until stable
* **Trade-off:** Higher utilization = more KV cache = higher throughput; Lower = more headroom for variable loads

== Reasoning ==

vLLM uses PagedAttention which dynamically allocates KV cache in blocks. The `gpu_memory_utilization` controls the fraction of GPU memory reserved for this cache after model weights are loaded. Memory layout:

1. **Model weights**: Fixed, loaded at startup
2. **KV cache**: Dynamic, sized based on `gpu_memory_utilization`
3. **Activation memory**: Small overhead for forward pass

If utilization is set too high, sudden spikes in batch size or sequence length can cause OOM. The 0.85-0.9 range provides a balance between throughput and stability.

== Code Evidence ==

From `vllm/engine/arg_utils.py` - Default batch token heuristic for A100:
<syntaxhighlight lang="python">
# NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
# runtime overhead, but may need adjustment for other GPUs.
# TODO(woosuk): Tune the default values for other hardware.
</syntaxhighlight>

Environment variable for memory allocation from `vllm/envs.py:76`:
<syntaxhighlight lang="python">
VLLM_MAIN_CUDA_VERSION: str = "12.9"
</syntaxhighlight>

GPU memory utilization is a key parameter in EngineArgs that controls KV cache sizing.

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_EngineArgs]]
* [[uses_heuristic::Implementation:vllm-project_vllm_LLM_init]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Basic_Offline_LLM_Inference]]
