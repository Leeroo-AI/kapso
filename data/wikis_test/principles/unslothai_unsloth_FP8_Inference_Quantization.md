# Principle: FP8 Inference Quantization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|NVIDIA FP8 Primer|https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html]]
* [[source::Paper|FP8 Formats for Deep Learning|https://arxiv.org/abs/2209.05433]]
* [[source::Blog|DeepSeek V3 Technical Report|https://github.com/deepseek-ai/DeepSeek-V3]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Quantization]], [[domain::Inference]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
8-bit floating point quantization technique that reduces memory footprint by 2x compared to FP16/BF16 while maintaining floating-point precision advantages for inference of very large language models.

=== Description ===
FP8 (8-bit floating point) quantization provides a middle ground between the severe precision loss of integer quantization and the memory requirements of 16-bit formats. Unlike INT8 which uses fixed-point arithmetic, FP8 retains the dynamic range benefits of floating-point representation.

Two FP8 formats are standardized:
1. **E4M3** (4 exponent, 3 mantissa bits) - Higher precision, narrower range, used for weights
2. **E5M2** (5 exponent, 2 mantissa bits) - Lower precision, wider range, used for gradients

**Key Characteristics:**
- 2x memory reduction vs FP16/BF16
- Dynamic range covers typical weight/activation distributions
- Hardware support on NVIDIA H100/H200 and Ada Lovelace GPUs
- Compatible with block-wise scaling for fine-grained precision control

**Problem Solved:**
Running extremely large models (100B+ parameters) requires aggressive memory optimization. While INT4/INT8 quantization works for inference, the training dynamics of FP8 better preserve model quality. Models like DeepSeek-V3 use FP8 natively to enable efficient training and deployment.

**Quantization Granularity:**
- **Per-tensor**: Single scale factor, simplest but least accurate
- **Per-row/channel**: One scale per row, good balance
- **Block-wise (128x128)**: Fine-grained scales, best accuracy

=== Usage ===
Use FP8 quantization when:
- Deploying models with 70B+ parameters for inference
- Using NVIDIA H100/H200 or Ada Lovelace GPUs with FP8 tensor cores
- Model was trained with FP8-aware training (e.g., DeepSeek-V3)
- Memory is constrained but INT4/INT8 quality loss is unacceptable

Do NOT use when:
- Running on older GPUs without FP8 hardware support
- Maximum quality is required (use BF16/FP16 instead)
- Model is small enough to fit in FP16

== Theoretical Basis ==
'''FP8 E4M3 Format:'''
<syntaxhighlight lang="python">
# FP8 E4M3 value representation
# 1 sign bit, 4 exponent bits, 3 mantissa bits

# Dynamic range: approximately ±448
# Precision: ~0.5% relative error

def fp8_quantize(tensor, block_size=128):
    """Block-wise FP8 quantization."""
    # Reshape into blocks
    blocks = tensor.reshape(-1, block_size)

    # Compute per-block scale (max absolute value / 448)
    scales = blocks.abs().max(dim=1, keepdim=True) / 448.0

    # Normalize and convert to FP8
    normalized = blocks / scales
    quantized = normalized.to(torch.float8_e4m3fn)

    return quantized, scales

def fp8_dequantize(quantized, scales):
    """Dequantize FP8 to higher precision for computation."""
    return quantized.to(torch.bfloat16) * scales
</syntaxhighlight>

'''Block-wise Quantization Benefits:'''
<math>
\text{Error}_{block} \approx O(\frac{1}{n_{block}}) \ll \text{Error}_{tensor} \approx O(\frac{1}{n_{total}})
</math>

With 128x128 blocks, quantization error is bounded by the local variation within each block rather than the global tensor range.

'''Memory Calculation:'''
For a 70B parameter model:
- FP16: 70B × 2 bytes = 140GB
- FP8: 70B × 1 byte + scales = ~72GB (with block scales)
- **~49% memory reduction**

'''FP8 Matmul Fusion:'''
<syntaxhighlight lang="python">
# Fused FP8 matmul avoids dequantization overhead
def fp8_matmul(A_fp8, B_fp8, A_scales, B_scales):
    """
    Compute A @ B where both inputs are FP8 quantized.
    Output is computed in FP32 accumulator, returned as BF16.
    """
    # Hardware performs: (A * A_scale) @ (B * B_scale)
    # Scales are factored into output: C = A @ B * (A_scale * B_scale)
    return hardware_fp8_gemm(A_fp8, B_fp8, A_scales, B_scales)
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FP8_Quantization]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Dtype_Selection]]
