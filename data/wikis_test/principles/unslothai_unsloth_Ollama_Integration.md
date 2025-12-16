{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Ollama Documentation|https://ollama.ai/]]
* [[source::Repo|Ollama GitHub|https://github.com/ollama/ollama]]
* [[source::Doc|Modelfile Reference|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
|-
! Domains
| [[domain::Model_Deployment]], [[domain::LLMs]], [[domain::DevOps]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Process of configuring exported GGUF models for deployment with Ollama, generating appropriate Modelfile templates that define prompt formatting, system messages, and inference parameters.

=== Description ===
Ollama integration involves creating a Modelfile that tells Ollama how to properly interact with a fine-tuned model. The Modelfile specifies:

1. **Model Source**: Path to the GGUF file
2. **Prompt Template**: Jinja-like template matching the model's training format
3. **Stop Tokens**: Sequences that signal generation completion
4. **Parameters**: Temperature, top_p, context length, and other inference settings
5. **System Prompt**: Default system message for the model

Proper template matching is critical - using Ollama's default template with a model fine-tuned on a different format causes degraded performance. Each model family (Llama-3, ChatML, Alpaca) requires a specific template configuration.

Unsloth includes 50+ pre-configured Ollama template mappings for common model architectures, automatically detecting the appropriate format during GGUF export.

=== Usage ===
Configure Ollama integration when:
- Deploying fine-tuned models for local inference
- Creating shareable model packages for team use
- Setting up development or testing environments
- Building applications that interact with local LLMs

The generated Modelfile enables:
- `ollama create mymodel -f Modelfile` to register the model
- `ollama run mymodel` for interactive chat
- API access via `localhost:11434`

== Theoretical Basis ==
Ollama Modelfiles use a domain-specific language to configure model behavior.

'''Modelfile Structure:'''
<syntaxhighlight lang="text">
# Basic Modelfile structure
FROM ./model-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|system_start|>{{ .System }}<|system_end|>{{ end }}
{{ if .Prompt }}<|user_start|>{{ .Prompt }}<|user_end|>{{ end }}
<|assistant_start|>{{ .Response }}<|assistant_end|>"""

PARAMETER stop "<|user_start|>"
PARAMETER stop "<|assistant_end|>"
PARAMETER temperature 0.7
PARAMETER num_ctx 4096

SYSTEM "You are a helpful AI assistant."
</syntaxhighlight>

'''Template Variables:'''
<syntaxhighlight lang="text">
Ollama Template Variables:
┌─────────────────────────────────────────────────────┐
│ Variable    │ Description                           │
├─────────────────────────────────────────────────────┤
│ .System     │ System message (if provided)          │
│ .Prompt     │ User's input message                  │
│ .Response   │ Model's generated response            │
│ .First      │ Boolean: true for first message       │
└─────────────────────────────────────────────────────┘
</syntaxhighlight>

'''Template Mapping for Common Formats:'''
<syntaxhighlight lang="python">
# Pseudo-code for Ollama template generation
OLLAMA_TEMPLATES = {
    "llama-3": '''{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>''',

    "chatml": '''{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>''',

    "alpaca": '''{{ if .System }}{{ .System }}

{{ end }}### Instruction:
{{ .Prompt }}

### Response:
{{ .Response }}''',
}

def generate_modelfile(gguf_path, model_type, system_prompt=None):
    """
    Generate Ollama Modelfile for a model.
    """
    template = OLLAMA_TEMPLATES[model_type]
    stop_tokens = get_stop_tokens(model_type)

    modelfile = f'FROM {gguf_path}\n\n'
    modelfile += f'TEMPLATE """{template}"""\n\n'

    for stop in stop_tokens:
        modelfile += f'PARAMETER stop "{stop}"\n'

    if system_prompt:
        modelfile += f'\nSYSTEM "{system_prompt}"\n'

    return modelfile
</syntaxhighlight>

'''Stop Token Configuration:'''
<syntaxhighlight lang="python">
# Pseudo-code for stop token mapping
STOP_TOKENS = {
    "llama-3": [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ],
    "chatml": [
        "<|im_start|>",
        "<|im_end|>",
    ],
    "alpaca": [
        "### Instruction:",
        "### Response:",
    ],
}
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Tips and Tricks ===
