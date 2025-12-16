{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|GGUF Format|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Testing]], [[domain::GGUF]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of verifying that exported GGUF files are valid and produce correct outputs before deployment.

=== Description ===

GGUF validation ensures:
- File is structurally valid
- Model metadata is correct
- Weights are properly quantized
- Model produces reasonable outputs

== Practical Guide ==

=== Check File Properties ===
<syntaxhighlight lang="python">
import os

gguf_path = "gguf_output/unsloth.Q4_K_M.gguf"

# Check file exists and size
if os.path.exists(gguf_path):
    size_gb = os.path.getsize(gguf_path) / (1024**3)
    print(f"GGUF file size: {size_gb:.2f} GB")
</syntaxhighlight>

=== Test with llama-cli ===
<syntaxhighlight lang="bash">
# Quick inference test
llama-cli -m gguf_output/unsloth.Q4_K_M.gguf \
    -p "Hello, how are you?" \
    -n 50

# Check model info
llama-cli -m gguf_output/unsloth.Q4_K_M.gguf --show-info
</syntaxhighlight>

=== Test with llama-cpp-python ===
<syntaxhighlight lang="python">
from llama_cpp import Llama

# Load and test
llm = Llama(
    model_path="gguf_output/unsloth.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # GPU acceleration
)

output = llm("Hello, how are you?", max_tokens=50)
print(output["choices"][0]["text"])
</syntaxhighlight>

=== Verify with Ollama ===
<syntaxhighlight lang="bash">
# Create and test model
ollama create mymodel -f Modelfile
ollama run mymodel "Hello, how are you?"
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GGUF_Export]]
