{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8(): 8-bit Matrix Multiplication|https://arxiv.org/abs/2208.07339]]
* [[source::Blog|Unsloth Documentation|https://docs.unsloth.ai]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Technique for loading Large Language Models with memory-efficient quantization while preserving model quality through 4-bit NormalFloat (NF4) representation and automatic optimization dispatch.

=== Description ===

Model Loading in Unsloth implements the QLoRA approach to loading quantized models, enabling training of large models on consumer hardware. The key innovations include:

**4-bit NormalFloat (NF4) Quantization:**
- Weights are stored in 4-bit format using an information-theoretically optimal data type
- Double quantization further compresses quantization constants
- Paged optimizers prevent memory spikes during training

**Automatic Architecture Dispatch:**
- Model type is detected from config (Llama, Mistral, Gemma, Qwen, etc.)
- Architecture-specific optimizations are applied automatically
- Attention backend is selected based on available hardware (Flash Attention, xformers, SDPA)

**Memory Optimization Stack:**
- Sequential device mapping for multi-GPU setups
- Gradient checkpointing reduces activation memory by 30%
- Mixed precision training with automatic dtype selection

This approach solves the fundamental challenge of training 7B+ parameter models on GPUs with limited VRAM (8-24GB) while maintaining training quality comparable to full-precision fine-tuning.

=== Usage ===

Use this principle when:
* You need to fine-tune models larger than your GPU's VRAM can hold in full precision
* You want faster training without sacrificing model quality
* You're working with consumer GPUs (RTX 3090, 4090, etc.) or cloud instances (T4, A10G)

The 4-bit quantization provides a 4x memory reduction with minimal quality loss, enabling:
* 7B models on 8GB GPUs
* 13B models on 16GB GPUs
* 70B models on 48GB GPUs (with gradient checkpointing)

== Theoretical Basis ==

=== 4-bit NormalFloat (NF4) ===

NF4 is an information-theoretically optimal quantization data type for normally distributed weights:

<math>
W_{quantized} = \text{round}\left(\frac{W - W_{min}}{W_{max} - W_{min}} \cdot (2^4 - 1)\right)
</math>

The NF4 quantile values are precomputed to minimize quantization error for normally distributed data:

'''Pseudo-code:'''
<syntaxhighlight lang="python">
# NF4 quantization levels (16 values for 4-bit)
nf4_quantiles = [-1.0, -0.6961928, -0.5250730, -0.3949338,
                 -0.2844316, -0.1848489, -0.0911179,  0.0,
                  0.0796081,  0.1609130,  0.2461123,  0.3379513,
                  0.4407166,  0.5626170,  0.7229568,  1.0]

# Weight quantization
def quantize_nf4(weights):
    # Normalize to [-1, 1]
    absmax = weights.abs().max()
    normalized = weights / absmax

    # Find nearest NF4 quantile
    indices = find_nearest(normalized, nf4_quantiles)

    return indices, absmax  # Store indices + scale
</syntaxhighlight>

=== Double Quantization ===

To further reduce memory, quantization constants themselves are quantized:

<math>
\text{Memory} = \frac{\text{params}}{2} + \frac{\text{params}}{64} \cdot 32 \approx 0.5 + 0.5 = 1 \text{ byte/param}
</math>

This yields ~4.5 bits per parameter in practice (vs 4 bits theoretical).

=== Architecture Dispatch ===

'''Model type detection and optimization selection:'''
<syntaxhighlight lang="python">
# Abstract dispatch logic
def dispatch_model(model_type, config):
    optimizations = {
        "llama": LlamaOptimizations,
        "mistral": MistralSlidingWindow,
        "gemma2": Gemma2Softcapping,
        "qwen": QwenQKNormalization,
    }

    # Apply architecture-specific kernels
    optimizer = optimizations.get(model_type, GenericOptimizer)

    # Select attention backend
    if has_flash_attention():
        attention = FlashAttention2
    elif has_xformers():
        attention = XFormersAttention
    else:
        attention = SDPAttention

    return optimizer, attention
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Memory_Optimization]]
