# Implementation: unslothai_unsloth_ollama_modelfile

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Ollama|https://ollama.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for generating Ollama Modelfiles alongside GGUF exports.

=== Description ===

Unsloth automatically generates Ollama Modelfiles when saving to GGUF. The Modelfile specifies:
- Model location (GGUF file)
- Chat template (matched to model family)
- Sampling parameters
- System prompt

This enables one-command import into Ollama.

=== Usage ===

Automatically generated during `save_pretrained_gguf`. Use for Ollama deployment.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/ollama_template_mappers.py
* '''Lines:''' L1-2192

=== Generated Modelfile Example ===
<syntaxhighlight lang="text">
FROM ./model-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.7
</syntaxhighlight>

=== Importing to Ollama ===
<syntaxhighlight lang="bash">
# Import model
ollama create mymodel -f ./Modelfile

# Run
ollama run mymodel "Hello!"
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Ollama_Export]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
