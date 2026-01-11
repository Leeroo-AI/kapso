# Implementation: ALLOWED_QUANTS

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Quantization]], [[domain::GGUF]], [[domain::Model_Compression]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete reference for available GGUF quantization methods in Unsloth.

=== Description ===

`ALLOWED_QUANTS` is a dictionary mapping quantization method names to human-readable descriptions. It defines all valid quantization options for `save_pretrained_gguf` and `push_to_hub_gguf`.

=== Usage ===

Reference this when selecting quantization_method parameter for GGUF export.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L104-131

=== Definition ===
<syntaxhighlight lang="python">
ALLOWED_QUANTS = {
    "not_quantized": "Recommended. Fast conversion. Slow inference, big files.",
    "fast_quantized": "Recommended. Fast conversion. OK inference, OK file size.",
    "quantized": "Recommended. Slow conversion. Fast inference, small files.",
    "f32": "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "bf16": "Bfloat16 - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "f16": "Float16  - Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0": "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m": "Recommended. Uses Q6_K for half of attention.wv and feed_forward.w2, else Q4_K",
    "q5_k_m": "Recommended. Uses Q6_K for half of attention.wv and feed_forward.w2, else Q5_K",
    "q2_k": "Uses Q4_K for attention.vw and feed_forward.w2, Q2_K for other tensors.",
    "q3_k_l": "Uses Q5_K for attention.wv, attention.wo, feed_forward.w2, else Q3_K",
    "q3_k_m": "Uses Q4_K for attention.wv, attention.wo, feed_forward.w2, else Q3_K",
    "q3_k_s": "Uses Q3_K for all tensors",
    "q4_0": "Original quant method, 4-bit.",
    "q4_1": "Higher accuracy than q4_0 but not as high as q5_0. Quicker inference than q5.",
    "q4_k_s": "Uses Q4_K for all tensors",
    "q4_k": "alias for q4_k_m",
    "q5_k": "alias for q5_k_m",
    "q5_0": "Higher accuracy, higher resource usage and slower inference.",
    "q5_1": "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s": "Uses Q5_K for all tensors",
    "q6_k": "Uses Q8_K for all tensors",
    "q3_k_xs": "3-bit extra small quantization",
}
</syntaxhighlight>

== I/O Contract ==

=== Alias Resolution ===
{| class="wikitable"
|-
! Alias !! Resolves To !! Description
|-
| "not_quantized" || "f16" or "bf16" || Full precision based on model dtype
|-
| "fast_quantized" || "q8_0" || Quick conversion, good quality
|-
| "quantized" || "q4_k_m" || Best balance of size/quality
|-
| "q4_k" || "q4_k_m" || Alias
|-
| "q5_k" || "q5_k_m" || Alias
|}

== Usage Examples ==

=== List Available Methods ===
<syntaxhighlight lang="python">
from unsloth.save import ALLOWED_QUANTS, print_quantization_methods

# Print all available methods with descriptions
print_quantization_methods()

# Access programmatically
for method, description in ALLOWED_QUANTS.items():
    print(f"{method}: {description}")
</syntaxhighlight>

=== Select Method for Export ===
<syntaxhighlight lang="python">
# Recommended for most use cases
quantization_method = "q4_k_m"  # 4-bit with mixed precision

# For higher quality
quantization_method = "q8_0"  # 8-bit

# Multiple formats at once
quantization_method = ["q4_k_m", "q8_0", "q5_k_m"]
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Quantization_Selection]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

