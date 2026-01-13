# Implementation: ALLOWED_QUANTS

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|llama.cpp|https://github.com/ggerganov/llama.cpp]]
|-
! Domains
| [[domain::Quantization]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete reference for supported GGUF quantization methods in Unsloth.

=== Description ===

`ALLOWED_QUANTS` is a dictionary in unsloth/save.py that defines all supported quantization methods for GGUF conversion. It maps quantization identifiers to human-readable descriptions and validates user-provided quantization choices.

=== Usage ===

Reference this when choosing quantization_method for save_pretrained_gguf or push_to_hub_gguf. Use the string keys as values for the quantization_method parameter.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' 104-131

=== Definition ===
<syntaxhighlight lang="python">
ALLOWED_QUANTS = {
    # Aliases (Recommended)
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",

    # Full precision
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "f16": "Float16  - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",

    # 8-bit
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",

    # K-quants (Mixed precision - Recommended)
    "q4_k_m": "Recommended. Uses Q6_K for half of attention.wv and feed_forward.w2, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of attention.wv and feed_forward.w2, else Q5_K",
    "q6_k": "Uses Q8_K for all tensors",

    # Lower precision K-quants
    "q2_k": "Uses Q4_K for attention.vw and feed_forward.w2, Q2_K for others.",
    "q3_k_l": "Uses Q5_K for attention.wv, attention.wo, feed_forward.w2, else Q3_K",
    "q3_k_m": "Uses Q4_K for attention.wv, attention.wo, feed_forward.w2, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",

    # Standard quants
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. Quicker inference than q5.",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q4_k": "alias for q4_k_m",
    "q5_k": "alias for q5_k_m",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s": "Uses Q5_K for all tensors",

    # Extra small
    "q3_k_xs": "3-bit extra small quantization",
}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth.save import ALLOWED_QUANTS, print_quantization_methods

# Print all available methods
print_quantization_methods()
</syntaxhighlight>

== I/O Contract ==

=== Usage ===
{| class="wikitable"
|-
! Key !! Maps To !! Description
|-
| not_quantized || f16/bf16 || Full precision, no quantization
|-
| fast_quantized || q8_0 || Quick conversion, 8-bit
|-
| quantized || q4_k_m || Balanced, 4-bit mixed precision
|}

== Usage Examples ==

=== Using Aliases ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)

# Use friendly aliases
model.save_pretrained_gguf(
    "./output",
    tokenizer,
    quantization_method="fast_quantized",  # Maps to q8_0
)

model.save_pretrained_gguf(
    "./output",
    tokenizer,
    quantization_method="quantized",  # Maps to q4_k_m
)
</syntaxhighlight>

=== Using Specific Methods ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(...)

# Use specific quantization methods
model.save_pretrained_gguf(
    "./output",
    tokenizer,
    quantization_method="q5_k_m",  # Specific method
)
</syntaxhighlight>

=== Listing Available Methods ===
<syntaxhighlight lang="python">
from unsloth.save import print_quantization_methods

# Print all available quantization methods
print_quantization_methods()

# Output:
# "not_quantized"  ==> Recommended. Fast conversion. Slow inference, big files.
# "fast_quantized" ==> Recommended. Fast conversion. OK inference, OK file size.
# "quantized"      ==> Recommended. Slow conversion. Fast inference, small files.
# "f32"            ==> Not recommended. Retains 100% accuracy, but super slow...
# ...
</syntaxhighlight>

== Quantization Comparison ==

{| class="wikitable"
|-
! Method !! Bits !! Size (7B) !! Quality !! Speed
|-
| bf16/f16 || 16 || ~14 GB || 100% || Slow
|-
| q8_0 || 8 || ~7 GB || ~99.5% || Medium
|-
| q6_k || 6.5 || ~5.5 GB || ~99% || Medium-Fast
|-
| q5_k_m || 5.5 || ~5 GB || ~98% || Fast
|-
| q4_k_m || 4.5 || ~4 GB || ~97% || Fast
|-
| q4_0 || 4 || ~3.5 GB || ~95% || Very Fast
|-
| q3_k_m || 3.5 || ~3 GB || ~93% || Very Fast
|-
| q2_k || 2.5 || ~2.5 GB || ~90% || Very Fast
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Quantization_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_Ollama]]
