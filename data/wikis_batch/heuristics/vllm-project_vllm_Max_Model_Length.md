# Heuristic: vllm-project_vllm_Max_Model_Length

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

Context length tuning to balance memory usage and maximum sequence support for different use cases.

=== Description ===

The `max_model_len` parameter sets the maximum context length (input + output tokens) that vLLM will support. Longer contexts require more KV cache memory. Setting this appropriately prevents OOM errors while maximizing usable context for your application.

=== Usage ===

Use this heuristic when:
- Encountering **OOM errors** with long sequences
- Need to **limit context length** for specific applications
- Running models with **very long native context** (128K+) on limited VRAM
- Optimizing for **throughput** vs **context length** trade-off

== The Insight (Rule of Thumb) ==

* **Action:** Set `max_model_len` in EngineArgs or CLI
* **Default:** Uses model's native max context length
* **Memory Rule:** KV cache ≈ 2 * num_layers * hidden_dim * max_len * batch_size * bytes_per_element
* **Common Values:**
  - 2048 for constrained memory
  - 4096 for general chat
  - 8192-16384 for document analysis
  - 32768+ for RAG applications
* **Trade-off:** Longer max length = larger KV cache = fewer concurrent requests

== Reasoning ==

KV cache memory scales linearly with `max_model_len`. For models with 128K+ native context, using the full context on a single GPU may be impossible. Reducing `max_model_len` to your actual needs frees memory for larger batch sizes and higher throughput.

Memory calculation for Llama-70B at different lengths:
- 4K context: ~8GB KV cache per GPU (TP=4)
- 16K context: ~32GB KV cache per GPU (TP=4)
- 128K context: ~256GB KV cache per GPU (TP=4) - requires H100 cluster

== Code Evidence ==

Max model length verification from `vllm/config/vllm.py:1220-1224`:
<syntaxhighlight lang="python">
def recalculate_max_model_len(self, max_model_len: int):
    # Can only be called in try_verify_and_update_config
    model_config = self.model_config
    max_model_len = model_config.get_and_verify_max_len(max_model_len)
    self.model_config.max_model_len = max_model_len
</syntaxhighlight>

Logging max sequence length from `vllm/config/vllm.py:1305`:
<syntaxhighlight lang="python">
f"max_seq_len={self.model_config.max_model_len}, "
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_EngineArgs]]
* [[uses_heuristic::Implementation:vllm-project_vllm_LLM_init]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Basic_Offline_LLM_Inference]]
