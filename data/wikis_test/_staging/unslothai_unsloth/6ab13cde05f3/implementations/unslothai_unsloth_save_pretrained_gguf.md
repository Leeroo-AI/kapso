{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|GGUF Export Guide|https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf]]
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Quantization]], [[domain::GGUF]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Concrete tool for converting fine-tuned models to GGUF format with optional quantization for deployment with llama.cpp, Ollama, and compatible inference engines.

=== Description ===

`save_pretrained_gguf` is a method dynamically attached to models by Unsloth that handles the complete GGUF export pipeline:

* **Automatic llama.cpp installation**: Downloads and builds llama.cpp if not present
* **LoRA weight merging**: Combines adapters with base weights before conversion
* **GGUF conversion**: Uses llama.cpp's convert scripts to produce GGUF files
* **Quantization**: Applies various quantization methods (q4_k_m, q5_k_m, q8_0, etc.)
* **Multiple outputs**: Can generate multiple quantization levels in one call

The method handles architecture-specific conversion requirements including:
- Tokenizer format detection (BPE vs SentencePiece)
- Vocabulary fixes for special tokens
- Chat template mapping for Ollama compatibility

=== Usage ===

Call `model.save_pretrained_gguf()` to export your trained model to GGUF format. This is typically the final deployment step for llama.cpp or Ollama.

Use cases:
* Deploying models with Ollama for local inference
* Running models with llama.cpp on CPU or consumer GPUs
* Creating portable model files for distribution

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/save.py unsloth/save.py]
* '''Lines:''' 1776-2000 (unsloth_save_pretrained_gguf), 1061-1200 (save_to_gguf)

=== Signature ===
<syntaxhighlight lang="python">
def save_pretrained_gguf(
    self,
    save_directory: Union[str, os.PathLike],
    tokenizer = None,
    quantization_method: Union[str, List[str]] = "fast_quantized",
    first_conversion: str = None,  # "f16", "bf16", "f32"
    push_to_hub: bool = False,
    token: Optional[str] = None,
    private: Optional[bool] = None,
    # llama.cpp settings
    llama_cpp_path: str = None,
    maximum_memory_usage: float = 0.9,
):
    """
    Convert model to GGUF format with quantization.

    Args:
        save_directory: Output directory for GGUF files
        tokenizer: Tokenizer for vocabulary export
        quantization_method: Single method or list of methods:
            - "not_quantized" / "f16" / "bf16": Full precision
            - "fast_quantized" / "q8_0": 8-bit (fast conversion)
            - "quantized" / "q4_k_m": 4-bit (recommended)
            - "q5_k_m", "q3_k_m", "q2_k", etc.
        first_conversion: Initial precision before quantization
        push_to_hub: Use push_to_hub_gguf() instead
        llama_cpp_path: Custom llama.cpp installation path
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# save_pretrained_gguf is a method attached to the model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| save_directory || str || Yes || Output directory for GGUF files
|-
| tokenizer || PreTrainedTokenizer || No || Tokenizer (recommended for proper vocab export)
|-
| quantization_method || str or List[str] || No || Quantization type(s) (default: "fast_quantized")
|-
| first_conversion || str || No || Initial precision: "f16", "bf16", or "f32"
|-
| maximum_memory_usage || float || No || GPU memory threshold during merge (default: 0.9)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| GGUF files || Files || Named as {save_directory}/unsloth.{QUANT_METHOD}.gguf
|-
| Return || Tuple[List[str], bool, bool] || (file_locations, want_full_precision, is_vlm)
|}

== Usage Examples ==

=== Basic GGUF Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# After training...
# Export to GGUF with recommended quantization
model.save_pretrained_gguf(
    "gguf_output",
    tokenizer,
    quantization_method="q4_k_m",  # Recommended balance of speed/quality
)

# Creates: gguf_output/unsloth.Q4_K_M.gguf
</syntaxhighlight>

=== Multiple Quantization Methods ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Generate multiple GGUF files at different quantization levels
model.save_pretrained_gguf(
    "gguf_output",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m", "q8_0"],
)

# Creates:
# gguf_output/unsloth.Q4_K_M.gguf  (smallest, fastest)
# gguf_output/unsloth.Q5_K_M.gguf  (balanced)
# gguf_output/unsloth.Q8_0.gguf   (highest quality)
</syntaxhighlight>

=== Full Precision Export ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Export without quantization (for benchmarking or re-quantization)
model.save_pretrained_gguf(
    "gguf_output",
    tokenizer,
    quantization_method="f16",  # Full float16 precision
)

# Creates: gguf_output/unsloth.F16.gguf (large file, highest accuracy)
</syntaxhighlight>

=== Push to HuggingFace Hub ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Export GGUF and upload directly to Hub
model.push_to_hub_gguf(
    "your-username/my-model-GGUF",
    tokenizer,
    quantization_method=["q4_k_m", "q5_k_m"],
    token="hf_...",
)

# Files available at: https://huggingface.co/your-username/my-model-GGUF
</syntaxhighlight>

=== Deploy with Ollama ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Export GGUF
model.save_pretrained_gguf(
    "ollama_model",
    tokenizer,
    quantization_method="q4_k_m",
)

# Unsloth auto-generates Modelfile content based on model's chat template
# Manual Modelfile example:
modelfile = '''
FROM ./unsloth.Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
"""

PARAMETER stop "<|eot_id|>"
'''

# Deploy: ollama create mymodel -f Modelfile
# Run: ollama run mymodel
</syntaxhighlight>

== Quantization Methods Reference ==

{| class="wikitable"
|-
! Method !! BPW !! Quality !! Speed !! Description
|-
| f16 / bf16 || 16.0 || Best || Slowest || Full precision (reference)
|-
| q8_0 || 8.0 || Excellent || Good || Fast conversion, high quality
|-
| q5_k_m || ~5.5 || Very Good || Fast || Recommended for quality
|-
| q4_k_m || ~4.5 || Good || Very Fast || Recommended default
|-
| q3_k_m || ~3.5 || OK || Fastest || Resource-constrained
|-
| q2_k || ~2.5 || Basic || Fastest || Extreme compression
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_GGUF_Conversion]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_llama_cpp]]
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Quantization_Method_Selection]]
