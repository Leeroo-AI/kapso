# Principle: unslothai_unsloth_Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8(): 8-bit Matrix Multiplication|https://arxiv.org/abs/2208.07339]]
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Loading]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for loading large language models with 4-bit quantization to enable memory-efficient fine-tuning on consumer GPUs.

=== Description ===

Model Loading in the context of QLoRA fine-tuning involves loading pre-trained LLM weights in a compressed 4-bit format while maintaining the ability to compute gradients for fine-tuning. This is achieved through:

1. **NF4 Quantization**: Using the NormalFloat4 data type which is information-theoretically optimal for normally distributed weights
2. **Double Quantization**: Quantizing the quantization constants for additional memory savings
3. **Mixed Precision Compute**: Dequantizing to fp16/bf16 only during the forward/backward pass

The key insight from QLoRA is that 4-bit quantization preserves model quality while reducing memory by ~75%, enabling 7B+ models to fit on GPUs with <16GB VRAM.

=== Usage ===

Use this principle when:
- Fine-tuning large models (7B+) on consumer GPUs (RTX 3090, 4090)
- Memory is the primary constraint
- You're using LoRA adapters (not full fine-tuning)

This is the **first trainable step** after environment initialization in any QLoRA workflow.

== Theoretical Basis ==

=== NormalFloat4 (NF4) Quantization ===

The NF4 data type is designed specifically for neural network weights, which tend to be normally distributed:

<math>
\text{NF4 values} = \{-1.0, -0.6962, -0.5251, ..., 0, ..., 0.5251, 0.6962, 1.0\}
</math>

These 16 values are chosen to minimize quantization error for normal distributions:

<math>
E[\|W - Q(W)\|^2] \text{ is minimized for } W \sim \mathcal{N}(0, \sigma^2)
</math>

=== Memory Calculation ===

For a model with P parameters:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Memory comparison for 7B parameter model

# Full precision (fp32)
memory_fp32 = P * 4  # bytes = 28GB

# Half precision (fp16/bf16)
memory_fp16 = P * 2  # bytes = 14GB

# 4-bit quantization (NF4)
memory_4bit = P * 0.5 + overhead  # bytes = 3.5GB + ~1GB overhead

# QLoRA adds LoRA adapters (~1-2% of params)
memory_qlora = memory_4bit + (P * 0.02 * 2)  # bytes ≈ 4.5GB
</syntaxhighlight>

=== Dequantization During Compute ===

During forward/backward passes, weights are dequantized on-the-fly:

<syntaxhighlight lang="python">
# Abstract dequantization algorithm
def dequantize_nf4(quantized_weight, absmax, blocksize=64):
    # Each block of 64 values shares one absmax scaling factor
    blocks = split_into_blocks(quantized_weight, blocksize)

    for i, block in enumerate(blocks):
        # Map 4-bit indices to NF4 codebook values
        nf4_values = NF4_CODEBOOK[block]
        # Scale by block's absmax
        dequantized[i] = nf4_values * absmax[i]

    return dequantized.reshape(original_shape)
</syntaxhighlight>

=== Architecture Detection ===

The loader must identify the correct architecture to apply appropriate patches:

<syntaxhighlight lang="python">
# Model type detection
model_type = config.model_type  # "llama", "mistral", "qwen2", etc.

# Dispatch to architecture-specific optimizations
if model_type == "llama":
    dispatch_model = FastLlamaModel
elif model_type == "mistral":
    dispatch_model = FastMistralModel
elif model_type == "qwen2":
    dispatch_model = FastQwen2Model
# ... etc
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
