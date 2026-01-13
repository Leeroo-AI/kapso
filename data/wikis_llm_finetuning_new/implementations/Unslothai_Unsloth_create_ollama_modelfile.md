# Implementation: create_ollama_modelfile

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Ollama Modelfile|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
|-
! Domains
| [[domain::Deployment]], [[domain::Chat_Templates]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for generating Ollama Modelfiles from trained models provided by Unsloth.

=== Description ===

`create_ollama_modelfile` generates an Ollama-compatible Modelfile for a GGUF model. It:

1. Looks up the appropriate template from `OLLAMA_TEMPLATES` based on model name
2. Uses `MODEL_TO_OLLAMA_TEMPLATE_MAPPER` to map HuggingFace model names to template names
3. Substitutes the GGUF file location and EOS token
4. Returns a complete Modelfile string

The generated Modelfile is saved alongside the GGUF file during export.

=== Usage ===

This function is called automatically during `save_pretrained_gguf` and `push_to_hub_gguf`. The generated Modelfile is saved as `Modelfile` in the output directory.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' 1630-1683

=== Signature ===
<syntaxhighlight lang="python">
def create_ollama_modelfile(
    tokenizer: PreTrainedTokenizer,
    base_model_name: str,
    model_location: str,
) -> Optional[str]:
    """
    Creates an Ollama Modelfile for a GGUF model.

    Args:
        tokenizer: Tokenizer with EOS token information
        base_model_name: Original HuggingFace model name for template lookup
        model_location: Path to the GGUF file

    Returns:
        Modelfile string, or None if no template mapping exists
    """
</syntaxhighlight>

=== Related Files ===
* '''unsloth/ollama_template_mappers.py:''' Contains `OLLAMA_TEMPLATES` and `MODEL_TO_OLLAMA_TEMPLATE_MAPPER`

=== Import ===
<syntaxhighlight lang="python">
from unsloth.save import create_ollama_modelfile
from unsloth.ollama_template_mappers import OLLAMA_TEMPLATES, MODEL_TO_OLLAMA_TEMPLATE_MAPPER
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer with EOS token
|-
| base_model_name || str || Yes || Original model name for template lookup
|-
| model_location || str || Yes || Path to GGUF file (absolute or relative)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| modelfile || str or None || Complete Modelfile string, or None if no mapping
|}

== Usage Examples ==

=== Automatic Modelfile Generation ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=16)
# ... training ...

# Modelfile is generated automatically
model.save_pretrained_gguf(
    save_directory="./model_gguf",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
)

# Check the generated Modelfile
with open("./model_gguf/Modelfile") as f:
    print(f.read())
</syntaxhighlight>

=== Using with Ollama ===
<syntaxhighlight lang="bash">
# After GGUF export
cd ./model_gguf

# View the Modelfile
cat Modelfile

# Create Ollama model
ollama create my-llama -f Modelfile

# Test the model
ollama run my-llama "Hello, how are you?"

# List models
ollama list
</syntaxhighlight>

=== Example Modelfile Output ===
<syntaxhighlight lang="text">
FROM ./model.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
</syntaxhighlight>

=== Manual Modelfile Creation ===
<syntaxhighlight lang="python">
from unsloth.save import create_ollama_modelfile

# Create Modelfile manually
modelfile = create_ollama_modelfile(
    tokenizer=tokenizer,
    base_model_name="meta-llama/Llama-3.2-3B-Instruct",
    model_location="./model.Q4_K_M.gguf",
)

if modelfile:
    with open("Modelfile", "w") as f:
        f.write(modelfile)
    print("Modelfile created successfully")
else:
    print("No template mapping found for this model")
</syntaxhighlight>

== Supported Models ==

Models with Ollama template mappings include:

{| class="wikitable"
|-
! Model Family !! Template Name
|-
| Llama 3, 3.1, 3.2 || llama-3.1
|-
| Qwen 2, 2.5 || qwen2.5
|-
| Mistral, Mixtral || mistral
|-
| Gemma, Gemma 2 || gemma
|-
| Phi-3, Phi-3.5 || phi3
|-
| DeepSeek || deepseek
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Ollama_Template_Generation]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_Ollama]]
