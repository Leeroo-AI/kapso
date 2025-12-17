# Heuristic: vllm-project_vllm_Tensor_Parallelism

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Docs|https://docs.vllm.ai/en/latest/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Distributed_Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Multi-GPU inference optimization using tensor parallelism for models too large for single GPU.

=== Description ===

Tensor parallelism (TP) splits model weights across multiple GPUs, enabling inference of models larger than single GPU memory. vLLM supports TP with automatic weight distribution and NCCL communication. Requires GPUs with P2P connectivity (NVLink preferred) for optimal performance.

=== Usage ===

Use this heuristic when:
- Running **models too large for single GPU** (e.g., 70B+ models)
- Need to **reduce per-GPU memory usage** to fit larger batch sizes
- Using **multi-GPU servers** (DGX, multi-A100 setups)
- Enabling **sequence parallelism** for long context models

== The Insight (Rule of Thumb) ==

* **Action:** Set `tensor_parallel_size` in EngineArgs or CLI
* **Value:** Usually 2, 4, or 8 (must divide number of attention heads)
* **Rule:** Model must have num_heads divisible by TP size
* **Memory:** Memory per GPU ≈ (model_size / TP) + KV_cache
* **Performance:** Communication overhead ~10-20% vs single GPU
* **Trade-off:** More GPUs = less memory per GPU, but more communication overhead

== Reasoning ==

Tensor parallelism shards the weight matrices across GPUs. Each GPU computes a portion of the attention heads and MLP, then results are all-reduced. The TP size must divide the number of attention heads evenly, or certain head sizes will be filtered out during chunked prefill.

Common configurations:
- **7B models:** TP=1 (single A100-80GB) or TP=2 (dual A100-40GB)
- **13B models:** TP=2 (dual A100-80GB) or TP=4 (quad A100-40GB)
- **70B models:** TP=4 (quad A100-80GB) or TP=8 (8x A100-40GB)
- **405B models:** TP=8 (8x H100-80GB)

== Code Evidence ==

Tensor parallel size filtering for chunked prefill from `vllm/config/vllm.py:1003-1017`:
<syntaxhighlight lang="python">
removed_sizes = [
    size
    for size in possible_sizes
    if size % self.parallel_config.tensor_parallel_size != 0
]
if removed_sizes:
    logger.warning(
        "Removing chunked prefill sizes %s that are not divisible by "
        "tensor_parallel_size=%d because "
        "sequence parallelism is enabled",
        removed_sizes,
        self.parallel_config.tensor_parallel_size,
    )

possible_sizes = [
    size
    for size in possible_sizes
    if size % self.parallel_config.tensor_parallel_size == 0
]
</syntaxhighlight>

Sequence parallelism with TP from `vllm/config/vllm.py:1116-1118`:
<syntaxhighlight lang="python">
if (
    self.parallel_config.tensor_parallel_size > 1
    and self.compilation_config.pass_config.enable_sp
):
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_EngineArgs]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Online_API_Serving]]
