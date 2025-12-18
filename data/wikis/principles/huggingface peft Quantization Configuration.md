{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|Quantization Survey|https://arxiv.org/abs/2103.13630]]
|-
! Domains
| [[domain::Quantization]], [[domain::Memory_Efficiency]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for configuring 4-bit quantization to enable training of large language models on consumer hardware.

=== Description ===

Quantization Configuration sets up the numerical precision for model weights and computation. QLoRA uses 4-bit NF4 (Normal Float 4-bit) quantization specifically designed for normally distributed neural network weights. This reduces memory requirements by ~4x while maintaining model quality through careful quantization scheme design.

Key concepts:
* **NF4:** Information-theoretically optimal for normal distributions
* **Double quantization:** Quantizes the quantization constants for additional savings
* **Compute dtype:** Higher precision (bfloat16) for computations ensures stability

=== Usage ===

Apply this principle when setting up QLoRA training:
* **Standard QLoRA:** Use NF4 with bfloat16 compute dtype
* **Maximum savings:** Enable double quantization for ~0.4 extra bits/param
* **Older GPUs:** Use float16 compute dtype if bfloat16 unsupported

== Theoretical Basis ==

'''Normal Float 4-bit (NF4):'''

NF4 quantizes weights to 4-bit values optimized for normally distributed data:

<math>Q_{NF4}(w) = \text{argmin}_{q_i \in Q} |w - q_i|</math>

Where <math>Q</math> contains 16 quantization levels positioned at quantiles of <math>\mathcal{N}(0,1)</math>.

'''Memory Calculation:'''

Base model memory with 4-bit quantization:
<math>\text{Memory}_{4bit} \approx \frac{P \times 4}{8} = \frac{P}{2} \text{ bytes}</math>

Where <math>P</math> is parameter count. For a 7B model:
<math>\text{Memory} \approx \frac{7 \times 10^9}{2} \approx 3.5 \text{ GB}</math>

'''Double Quantization:'''

Quantization uses scale factors per block. Double quantization quantizes these scales:
<syntaxhighlight lang="python">
# Pseudo-code for double quantization
block_size = 64
scales = compute_scales(weights, block_size)  # float16 scales
double_quant_scales = quantize_fp8(scales)    # Additional compression
</syntaxhighlight>

This saves ~0.4 bits per parameter (8 → 0.5 bytes for 64-element block scale).

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_BitsAndBytesConfig_4bit]]
