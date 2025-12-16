{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8(): 8-bit Matrix Multiplication|https://arxiv.org/abs/2208.07339]]
* [[source::Blog|BitsAndBytes 4-bit Quantization|https://huggingface.co/blog/4bit-transformers-bitsandbytes]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Quantization]], [[domain::Model_Loading]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Technique for loading pre-trained transformer models with reduced-precision quantization (4-bit NormalFloat) to dramatically reduce GPU memory requirements while preserving model quality.

=== Description ===
4-bit NormalFloat (NF4) quantization is a data type specifically designed for normally-distributed neural network weights. Unlike standard INT4 quantization which uses uniform bins, NF4 uses non-uniform quantization levels optimized for the Gaussian distribution typically found in transformer weights.

The key insight from QLoRA is that pre-trained model weights follow a zero-centered normal distribution with standard deviation σ. NF4 places quantization levels at the quantiles of this distribution, ensuring equal probability mass in each bin. This preserves more information than uniform quantization.

The loading process involves:
1. **Weight Quantization**: Convert FP16/BF16 weights to 4-bit NF4 representation
2. **Double Quantization**: Optionally quantize the quantization constants themselves for additional memory savings
3. **Compute Precision**: Maintain FP16/BF16 for compute while storing weights in 4-bit
4. **Architecture Detection**: Auto-detect model type (Llama, Mistral, Qwen, etc.) and apply appropriate optimizations

Memory savings are approximately 4x compared to FP16, enabling 7B models to fit in ~6GB VRAM and 70B models in ~40GB VRAM.

=== Usage ===
Use this technique when:
- Training or fine-tuning LLMs on consumer GPUs with limited VRAM (8-24GB)
- Loading large models (7B+) that would otherwise not fit in memory
- Performing QLoRA fine-tuning (4-bit base + 16-bit LoRA adapters)
- Inference on memory-constrained devices

Do not use when:
- Maximum accuracy is required (full-precision is marginally better)
- Performing full fine-tuning (all weights need to be trainable)
- The model fits comfortably in available VRAM

== Theoretical Basis ==
NF4 quantization maps continuous FP16 values to discrete 4-bit codes using a non-uniform mapping:

<math>
q_i = \Phi^{-1}\left(\frac{2i + 1}{32}\right) \cdot \sigma
</math>

Where:
- <math>q_i</math> is the i-th quantization level (i = 0 to 15)
- <math>\Phi^{-1}</math> is the inverse cumulative distribution function of the standard normal
- <math>\sigma</math> is the estimated standard deviation of the weight tensor

'''Quantization Process:'''
<syntaxhighlight lang="python">
# Pseudo-code for NF4 quantization
def quantize_nf4(weights):
    # Compute block-wise statistics (typically 64 weights per block)
    blocks = reshape(weights, [-1, block_size])
    scales = max(abs(blocks), axis=1)

    # Normalize to [-1, 1] range
    normalized = blocks / scales

    # Map to nearest NF4 quantile
    nf4_codes = find_nearest_quantile(normalized, NF4_LEVELS)

    return nf4_codes, scales
</syntaxhighlight>

'''Dequantization for Compute:'''
<syntaxhighlight lang="python">
# Pseudo-code for dequantization during forward pass
def dequantize_nf4(codes, scales):
    # Look up FP16 values from NF4 codebook
    values = NF4_CODEBOOK[codes]

    # Rescale to original magnitude
    return values * scales
</syntaxhighlight>

The 16 NF4 quantization levels (normalized) are approximately:
`[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]`

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Tips and Tricks ===
