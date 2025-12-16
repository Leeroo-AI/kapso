# Principle: QLoRA 4-bit Quantization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8(): 8-bit Matrix Multiplication|https://arxiv.org/abs/2208.07339]]
* [[source::Blog|Hugging Face QLoRA Guide|https://huggingface.co/blog/4bit-transformers-bitsandbytes]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Quantization]], [[domain::Memory_Optimization]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Memory-efficient quantization technique that enables fine-tuning of Large Language Models on consumer GPUs by storing model weights in 4-bit NormalFloat (NF4) format while computing in higher precision.

=== Description ===
QLoRA (Quantized Low-Rank Adaptation) combines two key innovations to dramatically reduce the memory footprint of LLM fine-tuning:

1. **4-bit NormalFloat (NF4) Quantization** - A novel data type that represents weights using only 4 bits per parameter, optimized for normally-distributed neural network weights. NF4 provides better information density than standard 4-bit integers.

2. **Double Quantization** - Quantizes the quantization constants themselves, further reducing memory by approximately 0.5 bits per parameter.

3. **Paged Optimizers** - Uses unified memory to prevent out-of-memory errors during gradient checkpointing by dynamically paging optimizer states between GPU and CPU.

The key insight is that while weights are stored in 4-bit precision, computations are performed in higher precision (bfloat16/float16) by dequantizing on-the-fly during the forward and backward passes. This preserves model quality while reducing memory by approximately 4x compared to full-precision training.

**Problem Solved:** Traditional full-precision fine-tuning of a 7B parameter model requires ~28GB VRAM. QLoRA reduces this to ~6GB, making fine-tuning accessible on consumer GPUs.

=== Usage ===
Use QLoRA 4-bit quantization when:
- Fine-tuning models larger than available GPU VRAM can hold in full precision
- Training on consumer GPUs (RTX 3090, 4090, or cloud T4/A10)
- Memory efficiency is more important than maximum training speed
- You need to fine-tune models with 7B+ parameters on 16GB or less VRAM

Do NOT use when:
- Maximum training speed is critical (full precision is ~10-20% faster per step)
- Model quality requirements demand full precision throughout
- Deploying to inference where quantization-aware training is preferred

== Theoretical Basis ==
The NF4 data type is based on quantile quantization optimized for normally-distributed weights:

'''Quantile Quantization:'''
<syntaxhighlight lang="python">
# NF4 levels are computed from the quantiles of a standard normal distribution
# Creating 2^4 = 16 levels that optimally cover the weight distribution

# The quantization process:
def quantize_nf4(tensor, block_size=64):
    """Quantize tensor to NF4 format with block-wise scaling."""
    # 1. Divide tensor into blocks
    blocks = tensor.reshape(-1, block_size)

    # 2. Compute per-block scaling factor (absmax normalization)
    scales = blocks.abs().max(dim=1, keepdim=True)
    normalized = blocks / scales

    # 3. Map to nearest NF4 quantile
    # NF4 levels: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    #              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
    quantized = map_to_nearest_level(normalized, NF4_LEVELS)

    return quantized, scales

def dequantize_nf4(quantized, scales):
    """Dequantize during forward/backward pass."""
    return quantized * scales  # Returns bfloat16 for computation
</syntaxhighlight>

'''Memory Calculation:'''
<math>
Memory_{QLoRA} = \frac{Parameters \times 4}{8} + Adapters_{16bit} + Gradients_{16bit}
</math>

For a 7B model with LoRA rank 64:
- Base weights: 7B × 0.5 bytes = 3.5GB (4-bit)
- LoRA adapters: ~50M × 2 bytes = ~100MB (16-bit)
- Gradients/activations: ~2-4GB (with gradient checkpointing)
- **Total: ~6-8GB vs ~28GB for full precision**

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Tips and Tricks ===
