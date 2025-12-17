# Implementation: unslothai_unsloth_export_format_selection_pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Decision_Making]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Pattern documentation for selecting the appropriate export format based on deployment requirements.

=== Description ===

Export Format Selection is a decision pattern, not an API. It guides choosing between:
- **LoRA only**: Smallest, requires base model at runtime
- **Merged 16-bit**: Standalone, for HuggingFace/PyTorch inference
- **GGUF**: CPU inference, Ollama, llama.cpp

=== Decision Tree ===

<syntaxhighlight lang="python">
def select_export_format(requirements):
    if requirements.get("cpu_only"):
        return "gguf"
    elif requirements.get("ollama"):
        return "gguf"
    elif requirements.get("hf_inference"):
        return "merged_16bit"
    elif requirements.get("size_critical"):
        return "lora"
    else:
        return "merged_16bit"  # Default
</syntaxhighlight>

== Format Comparison ==

| Format | Size (7B) | Inference Engine | Standalone |
|--------|-----------|------------------|------------|
| LoRA | ~100MB | HF + base model | No |
| Merged 16-bit | ~14GB | HuggingFace | Yes |
| GGUF q4_k_m | ~4GB | llama.cpp, Ollama | Yes |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Export_Format_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
