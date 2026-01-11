# Principle: Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|LLM.int8()|https://arxiv.org/abs/2208.07339]]
* [[source::Blog|Hugging Face Quantization|https://huggingface.co/docs/transformers/quantization]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for loading pre-trained Large Language Models with memory-efficient quantization while preserving model quality for fine-tuning.

=== Description ===

Model Loading for QLoRA fine-tuning involves loading a pre-trained transformer model with 4-bit NF4 (Normal Float 4-bit) quantization. This approach reduces the memory footprint of large models by approximately 75% while maintaining sufficient precision for effective fine-tuning through Low-Rank Adaptation (LoRA).

The key insight from QLoRA is that base model weights can be aggressively quantized to 4-bit precision during loading, while LoRA adapter weights are trained in higher precision (float16/bfloat16). The quantized base weights serve as a frozen foundation, with only the small LoRA matrices being updated during training.

This principle addresses the memory bottleneck that prevents fine-tuning large models on consumer GPUs. A 7B parameter model that would normally require ~28GB of VRAM in float32 can be loaded in ~4GB with 4-bit quantization.

=== Usage ===

Apply this principle when:
* Fine-tuning models larger than your GPU memory in float16
* Using consumer GPUs (RTX 3090, RTX 4090, A10G) for LLM training
* Training with LoRA/QLoRA adapters
* Prioritizing memory efficiency over maximum throughput

Do NOT apply when:
* Full parameter fine-tuning is required (no LoRA)
* Maximum inference speed is critical (quantization adds dequantization overhead)
* Model accuracy requirements exceed QLoRA's capabilities

== Theoretical Basis ==

=== NF4 Quantization ===

Normal Float 4-bit (NF4) is an information-theoretically optimal quantization scheme for normally distributed data:

<math>
Q_{NF4}(x) = \text{round}\left(\frac{x - \min(x)}{\max(x) - \min(x)} \times 15\right)
</math>

NF4 uses a non-uniform quantization grid that matches the empirical distribution of neural network weights, which tend to follow a normal distribution.

=== Double Quantization ===

QLoRA further reduces memory by quantizing the quantization constants:

<math>
\text{Memory} = \frac{n \times 4}{8} + \frac{n}{64} \times 32 + \frac{n}{64 \times 256} \times 32
</math>

Where:
* First term: 4-bit weights
* Second term: FP32 quantization scales (one per 64 weights)
* Third term: 8-bit quantization of the scales themselves

=== Precision Hierarchy ===

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# QLoRA precision hierarchy (abstract)
base_weights = quantize_nf4(pretrained_weights)  # 4-bit frozen
lora_A = init_random(r, d_in, dtype=float16)     # 16-bit trainable
lora_B = init_zeros(d_out, r, dtype=float16)     # 16-bit trainable

# Forward pass
h = dequantize(base_weights) @ x + (lora_B @ lora_A) @ x * scale
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained]]

