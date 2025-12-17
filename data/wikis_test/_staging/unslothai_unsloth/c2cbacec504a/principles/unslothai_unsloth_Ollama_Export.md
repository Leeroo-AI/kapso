# Principle: unslothai_unsloth_Ollama_Export

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Ollama|https://ollama.ai]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Deployment]], [[domain::Local_Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for packaging GGUF models with Ollama-compatible Modelfiles for local deployment.

=== Description ===

Ollama Export creates a complete package for Ollama:
1. **GGUF model file**: Quantized weights
2. **Modelfile**: Configuration including chat template
3. **Template mapping**: Auto-detection of correct template for model family

This enables running fine-tuned models locally with a simple `ollama run` command.

=== Usage ===

Use when deploying models for local inference via Ollama.

== Modelfile Components ==

| Component | Purpose |
|-----------|---------|
| FROM | Points to GGUF file |
| TEMPLATE | Chat template in Ollama format |
| PARAMETER | Sampling settings |
| SYSTEM | Default system prompt |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_ollama_modelfile]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
