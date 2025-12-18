{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Transformers Quantization|https://huggingface.co/docs/transformers/main_classes/quantization]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8()|https://arxiv.org/abs/2208.07339]]
* [[source::Repo|HuggingFace Transformers|https://github.com/huggingface/transformers]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-18 00:00 GMT]]
|}

== Overview ==
Define quantization parameters through a configuration object that specifies precision reduction strategy, data types, and module-level control.

=== Description ===
Quantization configuration establishes the contract between user intent and model transformation. It encapsulates decisions about precision levels (4-bit, 8-bit), quantization schemes (FP4, NF4, INT8), computational data types, and module-specific exceptions. This principle separates quantization policy from implementation mechanics, allowing the same configuration to be serialized, transmitted, and applied consistently across different execution contexts.

The configuration acts as a declarative specification that can be validated before any model weights are touched. It addresses the tension between memory efficiency and numerical stability by providing fine-grained control over which modules to quantize, what precision to use for computations versus storage, and how to handle outliers in activation distributions.

=== Usage ===
Apply this principle when you need to:
* Reduce model memory footprint for deployment on resource-constrained devices
* Enable loading of large language models that exceed available GPU memory
* Trade precision for throughput in inference workloads
* Serialize quantization settings alongside model checkpoints
* Validate quantization parameters before expensive model loading operations

== Theoretical Basis ==

=== Quantization Schemes ===

'''INT8 Quantization (LLM.int8()):'''
Maps floating-point weights to 8-bit integers using vector-wise quantization with outlier detection.

<pre>
For weight matrix W with outlier threshold t:
1. Identify outliers: O = {w_ij | |w_ij| > t}
2. Separate matrices: W = W_normal + W_outlier
3. Quantize normal weights: W_normal_int8 = round(W_normal / scale) where scale = max(|W_normal|) / 127
4. Keep outliers in FP16: W_outlier remains high precision
5. During computation: output = (W_normal_int8 * scale) @ input + W_outlier @ input
</pre>

'''4-bit NF4 (Normal Float 4):'''
Quantizes to 4-bit values optimally distributed for normally-distributed weights (common in neural networks after training).

<pre>
NF4 quantization levels (16 values from -1 to 1):
q_i = Φ^(-1)((i + 0.5) / 16) for i = 0..15
where Φ^(-1) is the inverse CDF of N(0, 1)

Quantization: q = argmin_i |w - scale * q_i|
Dequantization: w_approx = scale * q_i
</pre>

'''4-bit FP4 (Float 4):'''
Uses 4-bit floating-point representation with 1 sign bit, 2 exponent bits, 1 mantissa bit.

<pre>
FP4 format: s e_1 e_0 m_0
Values: (-1)^s * 2^(e-1) * (1 + m/2) for e > 0
        (-1)^s * 2^(-1) * (m/2) for e = 0
</pre>

=== Double Quantization ===

Applies quantization recursively to the quantization constants themselves:

<pre>
Standard quantization:
w_int4 = quantize(w, scale_fp32)  # scale stored in FP32

Double quantization:
scale_int8 = quantize(scale_fp32, block_scale)
w_int4 = quantize(w, dequantize(scale_int8, block_scale))
# Reduces scale storage from 32 bits to 8 bits per block
</pre>

=== Configuration Parameters ===

<pre>
Configuration = {
    precision: {4bit, 8bit},
    data_type: {FP4, NF4, INT8},
    compute_dtype: torch.dtype,  # for intermediate computations
    storage_dtype: torch.dtype,   # for quantized weight storage
    double_quant: bool,           # apply nested quantization to scales
    threshold: float,             # outlier detection threshold for INT8
    skip_modules: [module_names]  # keep specified modules in full precision
}

Memory savings:
- FP32 → INT8: 4× reduction
- FP32 → FP4/NF4: 8× reduction
- With double quantization: additional ~0.5 bits saved per weight
</pre>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_transformers_BitsAndBytesConfig_setup]]
