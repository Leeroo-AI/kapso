# Implementation: unslothai_unsloth_save_pretrained_gguf

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|GGUF Format|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for converting trained models to GGUF format for deployment with llama.cpp, Ollama, and other CPU/edge inference engines.

=== Description ===

`model.save_pretrained_gguf` orchestrates the complete GGUF conversion pipeline:

1. **Saves merged 16-bit model** if not already saved
2. **Installs llama.cpp** if not present (compiles from source)
3. **Converts to GGUF** using llama.cpp's convert script
4. **Quantizes** to target precision (q4_k_m, q8_0, etc.)
5. **Generates Ollama Modelfile** if requested

This enables deployment on devices without GPUs or with limited VRAM.

=== Usage ===

Use this when:
- Deploying to CPU-only environments
- Using Ollama or llama.cpp for inference
- Creating quantized models for edge devices
- Reducing model size for distribution

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L800-1500

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_gguf(
    self,
    save_directory: str,
    tokenizer: PreTrainedTokenizer = None,
    quantization_method: Union[str, List[str]] = "q4_k_m",
    push_to_hub: bool = False,
    token: Optional[str] = None,
    private: Optional[bool] = None,
) -> str:
    """
    Convert and save model in GGUF format.

    Args:
        save_directory: Path to save GGUF file
        tokenizer: Tokenizer (needed for conversion)
        quantization_method: Quantization type(s) - see ALLOWED_QUANTS
        push_to_hub: Upload to HuggingFace Hub
        token: HuggingFace API token
        private: Make repo private

    Returns:
        Path to saved GGUF file
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
# model now has .save_pretrained_gguf() method
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Path or HuggingFace repo ID
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer for conversion
|-
| quantization_method || str or List[str] || No (default: "q4_k_m") || Target quantization(s)
|-
| push_to_hub || bool || No (default: False) || Upload to Hub
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| file_path || str || Path to generated GGUF file
|-
| GGUF files || files || model-{quant}.gguf files
|-
| Modelfile || file || Ollama Modelfile (optional)
|}

== Quantization Methods ==

| Method | Bits | Size (7B) | Quality | Speed |
|--------|------|-----------|---------|-------|
| `f16` | 16 | 14GB | Best | Slowest |
| `q8_0` | 8 | 7GB | Excellent | Fast |
| `q4_k_m` | 4 | 4GB | Very Good | Very Fast |
| `q4_k_s` | 4 | 3.5GB | Good | Very Fast |
| `q3_k_m` | 3 | 2.8GB | Acceptable | Fastest |

== Usage Examples ==

=== Basic GGUF Conversion ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
model.save_pretrained_gguf(
    "my_model_gguf",
    tokenizer = tokenizer,
    quantization_method = "q4_k_m",  # 4-bit, good quality
)

# Creates: my_model_gguf/model-q4_k_m.gguf
</syntaxhighlight>

=== Multiple Quantizations ===
<syntaxhighlight lang="python">
# Generate multiple quantized versions
model.save_pretrained_gguf(
    "my_model_gguf",
    tokenizer = tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
)

# Creates:
# my_model_gguf/model-q4_k_m.gguf
# my_model_gguf/model-q8_0.gguf
# my_model_gguf/model-q5_k_m.gguf
</syntaxhighlight>

=== Upload to HuggingFace Hub ===
<syntaxhighlight lang="python">
model.save_pretrained_gguf(
    "username/my-model-gguf",
    tokenizer = tokenizer,
    quantization_method = "q4_k_m",
    push_to_hub = True,
    token = "hf_your_token",
)
</syntaxhighlight>

=== With Ollama Integration ===
<syntaxhighlight lang="python">
# Save GGUF and generate Ollama Modelfile
model.save_pretrained_gguf(
    "my_model_gguf",
    tokenizer = tokenizer,
    quantization_method = "q4_k_m",
)

# Then import to Ollama:
# ollama create my-model -f my_model_gguf/Modelfile
</syntaxhighlight>

=== Conversion Timeline ===
<syntaxhighlight lang="python">
# Typical timing for 7B model:
# 1. Install llama.cpp: ~3 minutes (first time only)
# 2. Convert to GGUF: ~3 minutes
# 3. Quantize: ~10 minutes per method
# Total: ~16+ minutes

print("Starting GGUF conversion - this may take 15+ minutes...")
model.save_pretrained_gguf(
    "output",
    tokenizer = tokenizer,
    quantization_method = ["q4_k_m", "q8_0"],
)
print("Done!")
</syntaxhighlight>

== Testing GGUF Output ==

<syntaxhighlight lang="bash">
# Test with llama.cpp CLI
./llama-cli -m ./model-q4_k_m.gguf -p "Hello, how are you?" -n 50

# Test with Ollama
ollama create mymodel -f ./Modelfile
ollama run mymodel "Hello, how are you?"

# Test with Python llama-cpp-python
from llama_cpp import Llama
llm = Llama(model_path="./model-q4_k_m.gguf")
output = llm("Hello", max_tokens=50)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_GGUF_Conversion]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
