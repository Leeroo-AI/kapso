# Principle: Ollama_Template_Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Ollama Modelfile|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Deployment]], [[domain::Model_Serialization]], [[domain::Chat_Templates]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for generating Ollama Modelfiles that enable deployment of GGUF models with correct chat templates.

=== Description ===

Ollama Modelfiles define how Ollama should load and interact with a GGUF model. They specify:

1. **FROM**: Path to the GGUF file
2. **TEMPLATE**: Go-template for formatting conversations
3. **PARAMETER**: Model parameters (temperature, stop tokens)
4. **SYSTEM**: Default system prompt

Unsloth automatically generates appropriate Modelfiles based on the model architecture, converting HuggingFace Jinja templates to Ollama's Go template format.

=== Usage ===

Use this principle when:
* Deploying GGUF models to Ollama
* Need chat template to match the trained model
* Creating a complete deployment package
* Distributing models for local inference

This step is automatically performed during GGUF export.

== Theoretical Basis ==

'''Modelfile Structure:'''
<syntaxhighlight lang="text">
FROM /path/to/model.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
</syntaxhighlight>

'''Template Conversion:'''
<syntaxhighlight lang="text">
Jinja (HuggingFace) → Go Template (Ollama)

{{ message['role'] }}  →  {{ .Role }}
{{ message['content'] }}  →  {{ .Content }}
{% if system %}  →  {{ if .System }}
{% for message in messages %}  →  {{ range .Messages }}
</syntaxhighlight>

'''Model-Specific Templates:'''
Unsloth maintains mappings for popular models:
{| class="wikitable"
|-
! Model Family !! Template Style !! Stop Token
|-
| Llama 3/3.1/3.2 || llama-3.1 || <nowiki><|eot_id|></nowiki>
|-
| Qwen 2/2.5 || chatml || <nowiki><|im_end|></nowiki>
|-
| Mistral/Mixtral || mistral || </s>
|-
| Gemma 2 || gemma || <end_of_turn>
|-
| Phi-3 || phi-3 || <nowiki><|end|></nowiki>
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_create_ollama_modelfile]]

