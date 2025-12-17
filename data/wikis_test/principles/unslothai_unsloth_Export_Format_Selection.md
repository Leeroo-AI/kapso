# Principle: unslothai_unsloth_Export_Format_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Deployment]], [[domain::Decision_Making]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for choosing the optimal model export format based on deployment constraints.

=== Description ===

Export Format Selection considers:
1. **Target runtime**: PyTorch, llama.cpp, Ollama, etc.
2. **Hardware**: GPU vs CPU, memory constraints
3. **Distribution**: File size, dependencies
4. **Quality requirements**: Precision needs

This is a decision point, not an algorithm—the right choice depends on use case.

=== Usage ===

Evaluate requirements before choosing export format in any Model Export workflow.

== Decision Framework ==

| Requirement | Best Format |
|-------------|-------------|
| HuggingFace deployment | Merged 16-bit |
| Ollama/llama.cpp | GGUF |
| Minimal distribution | LoRA only |
| CPU inference | GGUF |
| Maximum quality | Merged 16-bit or GGUF q8_0 |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_export_format_selection_pattern]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
