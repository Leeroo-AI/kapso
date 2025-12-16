{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Ollama|https://github.com/ollama/ollama]]
* [[source::Doc|Ollama Modelfile|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
|-
! Domains
| [[domain::Deployment]], [[domain::Inference]], [[domain::Ollama]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of deploying GGUF models to Ollama for easy local inference with automatic chat template handling.

=== Description ===

Ollama integration enables:
- Simple model deployment with `ollama run`
- Automatic chat template mapping
- REST API for applications
- Model versioning and management

== Practical Guide ==

=== Create Modelfile ===
<syntaxhighlight lang="text">
# Modelfile for Llama 3 style models
FROM ./unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
"""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
</syntaxhighlight>

=== Deploy to Ollama ===
<syntaxhighlight lang="bash">
# Create model from Modelfile
cd gguf_output
ollama create mymodel -f Modelfile

# Run model
ollama run mymodel "Hello, how are you?"

# List models
ollama list

# Remove model
ollama rm mymodel
</syntaxhighlight>

=== Use Ollama API ===
<syntaxhighlight lang="python">
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mymodel",
        "prompt": "Hello, how are you?",
        "stream": False
    }
)
print(response.json()["response"])
</syntaxhighlight>

=== Unsloth Template Mapping ===
<syntaxhighlight lang="python">
# Unsloth auto-maps HF templates to Ollama format
from unsloth.ollama_template_mappers import MODEL_TO_OLLAMA_TEMPLATE_MAPPER

# Check if model has Ollama template
model_type = "llama"
if model_type in MODEL_TO_OLLAMA_TEMPLATE_MAPPER:
    ollama_template = MODEL_TO_OLLAMA_TEMPLATE_MAPPER[model_type]
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GGUF_Export]]
