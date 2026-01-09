# Principle: Ollama_Modelfile

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Ollama Modelfile|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
|-
! Domains
| [[domain::Deployment]], [[domain::Ollama]], [[domain::Chat_Templates]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Generation of Ollama Modelfiles that map chat templates to Ollama's template syntax for local deployment.

=== Description ===

Ollama Modelfile generation translates HuggingFace chat templates to Ollama format:

* **FROM directive**: Points to GGUF file location
* **TEMPLATE directive**: Defines prompt formatting with Go template syntax
* **PARAMETER directives**: Sets inference parameters (temperature, stop tokens)
* **SYSTEM directive**: Optional default system message

=== Usage ===

Generated automatically during GGUF export. Use with `ollama create` to register the model.

== Theoretical Basis ==

=== Template Translation ===

HuggingFace templates (Jinja2) must be converted to Ollama's Go template syntax:

| HuggingFace | Ollama |
|-------------|--------|
| `{{ message['content'] }}` | `{{ .Content }}` |
| `{% if system %}` | `{{ if .System }}` |
| `{% for message in messages %}` | `{{ range .Messages }}` |

=== Stop Tokens ===

Stop tokens prevent generation beyond expected bounds:
* EOS token from tokenizer
* Role delimiters (e.g., `<|im_end|>` for ChatML)
* Special tokens from chat template

=== Supported Templates ===

Pre-defined templates include:
* ChatML (chatml)
* Llama-3 (llama-3)
* Alpaca (alpaca)
* Mistral (mistral)
* Gemma (gemma)
* Phi-3 (phi-3)
* Zephyr (zephyr)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_OLLAMA_TEMPLATES]]

