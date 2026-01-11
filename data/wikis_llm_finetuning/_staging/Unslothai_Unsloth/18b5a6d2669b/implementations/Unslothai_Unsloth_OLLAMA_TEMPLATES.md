# Implementation: OLLAMA_TEMPLATES

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Deployment]], [[domain::Ollama]], [[domain::Chat_Templates]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete collection of Ollama Modelfile templates for different chat formats.

=== Description ===

`OLLAMA_TEMPLATES` is a dictionary mapping chat template names to Ollama Modelfile strings. Each template includes:
* FROM directive placeholder
* TEMPLATE with Go template syntax
* PARAMETER stop tokens
* Optional SYSTEM message

=== Usage ===

Referenced automatically during GGUF export to generate appropriate Modelfile.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/ollama_template_mappers.py
* '''Lines:''' L1-500

=== Exports ===
<syntaxhighlight lang="python">
__all__ = [
    "OLLAMA_TEMPLATES",
    "OLLAMA_TEMPLATE_TO_MODEL_MAPPER",
    "MODEL_TO_OLLAMA_TEMPLATE_MAPPER",
]

OLLAMA_TEMPLATES = {}  # Template name -> Modelfile string
</syntaxhighlight>

== I/O Contract ==

=== Template Structure ===
<syntaxhighlight lang="text">
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
</syntaxhighlight>

=== Placeholders ===
{| class="wikitable"
|-
! Placeholder !! Description
|-
| `{__FILE_LOCATION__}` || Replaced with GGUF file path
|-
| `{__EOS_TOKEN__}` || Replaced with model's EOS token
|}

== Usage Examples ==

=== ChatML Template ===
<syntaxhighlight lang="python">
chatml_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''
OLLAMA_TEMPLATES["chatml"] = chatml_ollama
</syntaxhighlight>

=== Mistral Template ===
<syntaxhighlight lang="python">
mistral_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
'''
OLLAMA_TEMPLATES["mistral"] = mistral_ollama
</syntaxhighlight>

=== Using Generated Modelfile ===
<syntaxhighlight lang="bash">
# After GGUF export, Modelfile is created in output directory
# Register with Ollama:
ollama create my-model -f ./Modelfile

# Run the model:
ollama run my-model
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Ollama_Modelfile]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

