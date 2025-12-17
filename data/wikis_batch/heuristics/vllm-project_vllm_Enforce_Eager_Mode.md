# Heuristic: vllm-project_vllm_Enforce_Eager_Mode

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Docs|https://docs.vllm.ai/en/latest/]]
|-
! Domains
| [[domain::LLMs]], [[domain::Debugging]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Debugging technique to disable CUDA graph compilation for troubleshooting model compatibility issues.

=== Description ===

The `enforce_eager` flag disables CUDA graph capture and compilation optimizations. This is useful when debugging model loading issues, testing new model architectures, or when CUDA graphs cause errors with certain operations. Enabling this reduces performance but improves compatibility.

=== Usage ===

Use this heuristic when:
- Encountering **CUDA graph compilation errors**
- **Testing new model architectures** not yet optimized for vLLM
- **Debugging custom attention implementations**
- Experiencing **memory fragmentation** issues with CUDA graphs
- Running models with **dynamic shapes** that don't work well with graphs

== The Insight (Rule of Thumb) ==

* **Action:** Set `enforce_eager=True` in EngineArgs or `--enforce-eager` in CLI
* **Default:** False (CUDA graphs enabled)
* **Debug Mode:** Also set optimization level to -O0 for maximum debugging info
* **Performance Impact:** ~20-40% slower throughput without CUDA graphs
* **Trade-off:** Stability/compatibility vs performance

== Reasoning ==

CUDA graphs pre-record a sequence of GPU operations and replay them without CPU overhead. However, graphs require static shapes and can't handle dynamic operations. Some models with unusual architectures or custom CUDA kernels may not be compatible with graph capture. Enabling eager mode disables this optimization and runs operations one-by-one.

When `enforce_eager=True`, vLLM also sets the optimization level to -O0, providing more verbose error messages and debugging information.

== Code Evidence ==

Enforce eager mode logging from `vllm/config/vllm.py:622`:
<syntaxhighlight lang="python">
logger.warning("Enforce eager set, overriding optimization level to -O0")
</syntaxhighlight>

Related cascade attention disable from `vllm/config/vllm.py:888`:
<syntaxhighlight lang="python">
logger.warning_once("Disabling cascade attention when DBO is enabled.")
</syntaxhighlight>

Compilation level interaction from `vllm/config/compilation.py:739`:
<syntaxhighlight lang="python">
logger.warning(
    "Compilation level is set to %s, overriding enforce_eager to True",
    level,
)
</syntaxhighlight>

== Related Pages ==

* [[uses_heuristic::Implementation:vllm-project_vllm_EngineArgs]]
* [[uses_heuristic::Workflow:vllm-project_vllm_Online_API_Serving]]
