{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|8-bit Matrix Multiplication|https://arxiv.org/abs/2208.07339]]
* [[source::Doc|HuggingFace Quantization|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Optimization]], [[domain::Memory_Management]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Technique to reduce model memory footprint by representing weights in lower precision formats (4-bit, 8-bit) while preserving model quality.

=== Description ===
Quantization reduces the numerical precision of model weights from standard 32-bit or 16-bit floating point to lower bit-widths. For LLMs, 4-bit quantization (QLoRA) reduces memory by ~75% with minimal quality loss. The key innovation is NormalFloat4 (NF4), a data type optimized for the normal distribution of neural network weights, providing better accuracy than uniform 4-bit quantization.

=== Usage ===
Use this principle when you need to fit large models into limited GPU memory. Essential for running 7B+ parameter models on consumer GPUs (16-24GB VRAM). Apply when memory is the bottleneck, not compute speed. Works in combination with LoRA for efficient fine-tuning.

== Theoretical Basis ==
Quantization maps continuous values to a discrete set of levels:

\[
Q(w) = \text{round}\left(\frac{w - z}{s}\right) \cdot s + z
\]

Where:
* \( s \) = scale factor
* \( z \) = zero point

'''NF4 Quantization (QLoRA):'''
NF4 assumes weights follow a normal distribution N(0, σ). It creates 16 quantization bins positioned optimally for this distribution:

<syntaxhighlight lang="python">
# Simplified NF4 concept
def nf4_quantize(tensor):
    # Normalize to unit variance
    normalized = tensor / tensor.abs().max()
    
    # Quantize to 4-bit values optimized for normal distribution
    # Uses pre-computed optimal bin positions for N(0,1)
    quantized = map_to_nearest_nf4_bin(normalized)
    
    return quantized, scale
</syntaxhighlight>

'''Double Quantization:'''
QLoRA also quantizes the quantization constants (scales), providing additional ~0.4 bits/parameter savings.

'''Memory Comparison:'''
{| class="wikitable"
! Precision !! Bits/Parameter !! 7B Model Size
|-
|| FP32 || 32 || 28 GB
|-
|| FP16/BF16 || 16 || 14 GB
|-
|| INT8 || 8 || 7 GB
|-
|| NF4 || 4 || 3.5 GB
|}

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:BitsAndBytes_4bit_Quantization]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:AdamW_8bit_Optimizer_Usage]]

