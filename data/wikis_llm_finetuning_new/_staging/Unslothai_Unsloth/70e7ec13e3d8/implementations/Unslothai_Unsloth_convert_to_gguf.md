# Implementation: convert_to_gguf

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|llama.cpp|https://github.com/ggerganov/llama.cpp]]
|-
! Domains
| [[domain::Model_Serialization]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

External tool documentation for converting models to GGUF format via llama.cpp integration.

=== Description ===

The GGUF conversion pipeline is orchestrated by `save_to_gguf` in unsloth/save.py, which:

1. Installs llama.cpp if not present
2. Calls `convert_to_gguf` (from unsloth_zoo.llama_cpp) to create initial GGUF
3. Calls `quantize_gguf` for further quantization
4. Generates Ollama Modelfile

This is an **External Tool Doc** because the actual conversion uses llama.cpp binaries.

=== Usage ===

Use the model's `save_pretrained_gguf` or `push_to_hub_gguf` methods rather than calling conversion functions directly. These methods handle the complete pipeline including merging, conversion, and quantization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' 1070-1335 (save_to_gguf orchestrator)

=== External Dependencies ===
* '''llama.cpp:''' Provides llama-quantize and convert-hf-to-gguf.py
* '''unsloth_zoo.llama_cpp:''' Wrapper functions (convert_to_gguf, quantize_gguf)

=== Signature ===
<syntaxhighlight lang="python">
def save_to_gguf(
    model_name: str,
    model_type: str,
    model_dtype: str,
    is_sentencepiece: bool = False,
    model_directory: str = "unsloth_finetuned_model",
    quantization_method = "fast_quantized",  # Can be a list!
    first_conversion: str = None,
    is_vlm: bool = False,
    is_gpt_oss: bool = False,
) -> Tuple[List[str], bool, bool]:
    """
    Orchestrate complete GGUF conversion.

    Args:
        model_name: Name for output files
        model_type: Model architecture type
        model_dtype: Source dtype ("float16" or "bfloat16")
        model_directory: Path to merged HF model
        quantization_method: Target quantization(s)
        first_conversion: Initial GGUF dtype ("f16", "bf16")
        is_vlm: Whether model is vision-language

    Returns:
        Tuple of (output_paths, want_full_precision, is_vlm)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Usually accessed through model methods
model.save_pretrained_gguf(save_directory, tokenizer, quantization_method="q4_k_m")

# Or push directly
model.push_to_hub_gguf("username/model", tokenizer, quantization_method="q4_k_m")
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory for GGUF files
|-
| tokenizer || PreTrainedTokenizer || Yes || Tokenizer for Modelfile generation
|-
| quantization_method || str/List[str] || No || Target quantization(s) (default: "fast_quantized")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| GGUF files || Files || Quantized model files (e.g., model.Q4_K_M.gguf)
|-
| Modelfile || File || Ollama Modelfile for deployment
|}

== Usage Examples ==

=== Basic GGUF Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load and train model
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, r=16)
# ... training ...

# Save as GGUF with q4_k_m quantization
model.save_pretrained_gguf(
    save_directory="./model_gguf",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
)

# Files created:
# - ./model_gguf/model.Q4_K_M.gguf
# - ./model_gguf/Modelfile
</syntaxhighlight>

=== Multiple Quantizations ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, r=16)
# ... training ...

# Export multiple quantizations at once
model.save_pretrained_gguf(
    save_directory="./model_gguf",
    tokenizer=tokenizer,
    quantization_method=["q4_k_m", "q5_k_m", "q8_0"],
)

# Files created:
# - ./model_gguf/model.Q4_K_M.gguf
# - ./model_gguf/model.Q5_K_M.gguf
# - ./model_gguf/model.Q8_0.gguf
</syntaxhighlight>

=== Push GGUF to HuggingFace ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, r=16)
# ... training ...

# Convert and upload to HuggingFace Hub
model.push_to_hub_gguf(
    repo_id="username/my-model-gguf",
    tokenizer=tokenizer,
    quantization_method=["q4_k_m", "q8_0"],
    token="hf_xxx",
)
</syntaxhighlight>

=== Use with Ollama ===
<syntaxhighlight lang="bash">
# After saving GGUF with Modelfile
cd ./model_gguf

# Create Ollama model
ollama create my-model -f Modelfile

# Run inference
ollama run my-model "Hello, how are you?"
</syntaxhighlight>

== Quantization Methods ==

{| class="wikitable"
|-
! Method !! Description !! Use Case
|-
| not_quantized || Full f16/bf16 precision || Maximum accuracy
|-
| fast_quantized || q8_0 (8-bit) || Balance of speed and quality
|-
| quantized || q4_k_m (4-bit) || Good compression with quality
|-
| q8_0 || 8-bit quantization || High quality, 2x compression
|-
| q4_k_m || Mixed 4/6-bit || Recommended for most uses
|-
| q5_k_m || Mixed 5/6-bit || Higher quality than q4
|-
| q2_k || 2-bit quantization || Maximum compression
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_GGUF_Conversion]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_Ollama]]
