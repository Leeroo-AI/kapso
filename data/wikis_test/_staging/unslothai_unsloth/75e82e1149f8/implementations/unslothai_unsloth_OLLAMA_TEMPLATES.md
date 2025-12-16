# Implementation: OLLAMA_TEMPLATES

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Ollama Modelfile|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
|-
! Domains
| [[domain::Model_Export]], [[domain::GGUF]], [[domain::Deployment]], [[domain::Templates]]
|-
! Last Updated
| [[last_updated::2025-12-15 19:00 GMT]]
|}

== Overview ==
Concrete data structure containing Ollama Modelfile templates for various chat formats provided by the Unsloth library.

=== Description ===
`OLLAMA_TEMPLATES` is a dictionary mapping chat template names to Ollama Modelfile template strings. These templates are used when converting models to GGUF format with `create_ollama_modelfile=True`.

The templates define:
1. **Base Model** - Where to load the GGUF weights from
2. **Template** - Go template for formatting conversations
3. **Parameters** - Model parameters (temperature, stop tokens, etc.)
4. **System Prompt** - Default system message

Supported templates include:
- `llama-3`: Meta's Llama 3/3.1/3.2 instruction format
- `chatml`: ChatML format (Qwen, many others)
- `mistral`: Mistral instruction format
- `gemma`: Google Gemma format
- `zephyr`: Zephyr/Hugging Chat format
- And many more model-specific templates

=== Usage ===
Use these templates when:
- Creating Ollama Modelfiles for GGUF models
- Understanding how different models format conversations
- Customizing Ollama deployment configurations

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai/unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/ollama_template_mappers.py unsloth/ollama_template_mappers.py]
* '''Lines:''' 1-500+

Source Files: unsloth/ollama_template_mappers.py:L1-L500

=== Signature ===
<syntaxhighlight lang="python">
# Main exports
__all__ = [
    "OLLAMA_TEMPLATES",
    "OLLAMA_TEMPLATE_TO_MODEL_MAPPER",
    "MODEL_TO_OLLAMA_TEMPLATE_MAPPER",
]

# Template dictionary
OLLAMA_TEMPLATES: Dict[str, str] = {
    "unsloth": unsloth_ollama,
    "zephyr": zephyr_ollama,
    "chatml": chatml_ollama,
    "mistral": mistral_ollama,
    "mistral_v03": mistral_v03_ollama,
    "mistral_small": mistral_small_ollama,
    "llama-3": llama3_ollama,
    "llama-3.1": llama31_ollama,
    "llama-3.2": llama32_ollama,
    "gemma": gemma_ollama,
    "gemma2": gemma2_ollama,
    "phi-3": phi3_ollama,
    "qwen2": qwen2_ollama,
    "qwen2.5": qwen25_ollama,
    # ... many more templates
}

# Mapping from model name patterns to template names
MODEL_TO_OLLAMA_TEMPLATE_MAPPER: Dict[str, str] = {
    "llama-3": "llama-3",
    "llama-3.1": "llama-3.1",
    "llama-3.2": "llama-3.2",
    "mistral": "mistral",
    "qwen2": "chatml",
    "gemma": "gemma",
    # ... patterns for auto-detection
}

# Reverse mapping
OLLAMA_TEMPLATE_TO_MODEL_MAPPER: Dict[str, List[str]] = {
    "llama-3": ["llama-3", "llama3"],
    "chatml": ["qwen", "yi", "deepseek"],
    # ... template to model family mapping
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.ollama_template_mappers import (
    OLLAMA_TEMPLATES,
    MODEL_TO_OLLAMA_TEMPLATE_MAPPER,
    OLLAMA_TEMPLATE_TO_MODEL_MAPPER,
)
</syntaxhighlight>

== I/O Contract ==

=== Structure ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| OLLAMA_TEMPLATES || Dict[str, str] || Template name → Modelfile content
|-
| MODEL_TO_OLLAMA_TEMPLATE_MAPPER || Dict[str, str] || Model pattern → Template name
|-
| OLLAMA_TEMPLATE_TO_MODEL_MAPPER || Dict[str, List[str]] || Template → Compatible models
|}

=== Template Placeholders ===
{| class="wikitable"
|-
! Placeholder !! Description
|-
| {__FILE_LOCATION__} || Path to GGUF file (replaced at generation)
|-
| {__EOS_TOKEN__} || Model's EOS token (replaced at generation)
|-
| {{ .System }} || User's system message
|-
| {{ .Prompt }} || User's input prompt
|-
| {{ .Response }} || Model's response
|}

== Usage Examples ==

=== View Available Templates ===
<syntaxhighlight lang="python">
from unsloth.ollama_template_mappers import OLLAMA_TEMPLATES

# List all available templates
print("Available Ollama templates:")
for name in sorted(OLLAMA_TEMPLATES.keys()):
    print(f"  - {name}")
</syntaxhighlight>

=== Inspect a Template ===
<syntaxhighlight lang="python">
from unsloth.ollama_template_mappers import OLLAMA_TEMPLATES

# View the Llama 3 template
print(OLLAMA_TEMPLATES["llama-3"])

# Output:
# FROM {__FILE_LOCATION__}
# TEMPLATE """<|begin_of_text|>{{ if .System }}<|start_header_id|>system<|end_header_id|>
# {{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
# {{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# {{ .Response }}<|eot_id|>"""
# PARAMETER stop "<|eot_id|>"
# PARAMETER temperature 1.5
# PARAMETER min_p 0.1
</syntaxhighlight>

=== Auto-detect Template for Model ===
<syntaxhighlight lang="python">
from unsloth.ollama_template_mappers import MODEL_TO_OLLAMA_TEMPLATE_MAPPER

def get_template_for_model(model_name: str) -> str:
    """Find the appropriate template for a model."""
    model_lower = model_name.lower()

    for pattern, template in MODEL_TO_OLLAMA_TEMPLATE_MAPPER.items():
        if pattern in model_lower:
            return template

    return "chatml"  # Default fallback

# Examples
print(get_template_for_model("meta-llama/Llama-3.2-1B-Instruct"))
# Output: "llama-3.2"

print(get_template_for_model("Qwen/Qwen2-7B-Instruct"))
# Output: "chatml"
</syntaxhighlight>

=== Generate Custom Modelfile ===
<syntaxhighlight lang="python">
from unsloth.ollama_template_mappers import OLLAMA_TEMPLATES

def create_modelfile(
    gguf_path: str,
    template_name: str = "llama-3",
    eos_token: str = "<|eot_id|>",
) -> str:
    """Create an Ollama Modelfile from template."""
    template = OLLAMA_TEMPLATES[template_name]

    modelfile = template.replace("{__FILE_LOCATION__}", gguf_path)
    modelfile = modelfile.replace("{__EOS_TOKEN__}", eos_token)

    return modelfile

# Create Modelfile
modelfile_content = create_modelfile(
    "./model-Q4_K_M.gguf",
    template_name="llama-3.2",
    eos_token="<|eot_id|>",
)

# Save it
with open("Modelfile", "w") as f:
    f.write(modelfile_content)
</syntaxhighlight>

=== Template Examples ===
<syntaxhighlight lang="python">
# ChatML template (Qwen, Yi, etc.)
chatml_template = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

# Llama 3 template
llama3_template = '''
FROM {__FILE_LOCATION__}
TEMPLATE """<|begin_of_text|>{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

# Zephyr template
zephyr_template = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}{__EOS_TOKEN__}
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}{__EOS_TOKEN__}
{{ end }}<|assistant|>
{{ .Response }}{__EOS_TOKEN__}"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''
</syntaxhighlight>

=== Use with GGUF Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
# ... training ...

# Auto-generates Modelfile with correct template
model.save_pretrained_gguf(
    "gguf_output",
    tokenizer,
    quantization_method="q4_k_m",
    create_ollama_modelfile=True,  # Uses auto-detected template
)

# Then use with Ollama:
# $ cd gguf_output
# $ ollama create my-model -f Modelfile
# $ ollama run my-model
</syntaxhighlight>

=== Available Templates Reference ===
<syntaxhighlight lang="python">
# Main templates in OLLAMA_TEMPLATES:
templates = {
    "unsloth":       "Default Unsloth format",
    "zephyr":        "Zephyr/HuggingChat format",
    "chatml":        "ChatML format (Qwen, Yi, DeepSeek)",
    "mistral":       "Mistral v0.1/v0.2 format",
    "mistral_v03":   "Mistral v0.3+ format",
    "mistral_small": "Mistral Small format",
    "llama-3":       "Llama 3 instruction format",
    "llama-3.1":     "Llama 3.1 instruction format",
    "llama-3.2":     "Llama 3.2 instruction format",
    "gemma":         "Gemma 1 format",
    "gemma2":        "Gemma 2 format",
    "phi-3":         "Microsoft Phi-3 format",
    "qwen2":         "Qwen2 format (ChatML-based)",
    "qwen2.5":       "Qwen2.5 format",
    "vicuna":        "Vicuna format",
    "alpaca":        "Alpaca instruction format",
}
</syntaxhighlight>

== Related Pages ==
=== Context & Requirements ===
* Used automatically during GGUF export with `create_ollama_modelfile=True`
* Templates follow Ollama's Go template syntax
* Requires Ollama for deployment

=== Tips and Tricks ===
* Template is auto-detected based on model name
* Use `chatml` as fallback for unknown models
* Temperature and min_p are pre-configured for good generation
* Stop tokens are set per template to prevent infinite generation
