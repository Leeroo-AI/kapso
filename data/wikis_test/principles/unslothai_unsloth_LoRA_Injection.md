{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|RSLoRA: Rank-Stabilized LoRA|https://arxiv.org/abs/2312.03732]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::NLP]], [[domain::Parameter_Efficient_Training]], [[domain::Low_Rank_Adaptation]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Parameter-efficient fine-tuning technique that adds small trainable low-rank matrices to frozen model weights, enabling adaptation of large language models with minimal memory overhead.

=== Description ===

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that:

**Core Innovation:**
- Freezes original model weights W₀
- Adds trainable low-rank decomposition: ΔW = BA where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ
- During training, only A and B are updated (r << min(d,k))

**Unsloth Optimizations:**
- Fused LoRA kernels combine adapter computation with base operations
- Optimized for dropout=0 and bias="none" configurations
- Custom Triton kernels for attention and MLP LoRA operations

**Memory Benefits:**
- For r=16 on a 7B model: ~40MB additional parameters vs 14GB full model
- Combined with 4-bit quantization (QLoRA): training 7B models on 8GB GPUs
- Gradient checkpointing further reduces activation memory

This technique solves the fundamental problem of fine-tuning large models on limited hardware while achieving results competitive with full fine-tuning.

=== Usage ===

Use LoRA injection when:
* You need to adapt a large pre-trained model to a specific task
* You have limited GPU memory (8-24GB VRAM)
* You want fast training (fewer parameters = faster updates)
* You need to store multiple fine-tuned versions efficiently (small adapter files)

Typical configurations:
* **r=8-16**: General instruction tuning, chat fine-tuning
* **r=32-64**: Complex reasoning tasks, reinforcement learning
* **r=128+**: Large-scale adaptation (approaching full fine-tuning capacity)

== Theoretical Basis ==

=== Low-Rank Decomposition ===

The key insight is that weight updates during fine-tuning have low "intrinsic rank":

<math>
W = W_0 + \Delta W = W_0 + BA
</math>

Where:
- W₀ ∈ ℝᵈˣᵏ: Original frozen weights
- B ∈ ℝᵈˣʳ: Down-projection matrix
- A ∈ ℝʳˣᵏ: Up-projection matrix
- r: Rank (typically 8-128, much smaller than d and k)

=== Forward Pass ===

During forward pass, the adapted layer computes:

<math>
h = W_0 x + \frac{\alpha}{r} BAx
</math>

Where α is the scaling factor (lora_alpha).

'''Pseudo-code:'''
<syntaxhighlight lang="python">
# LoRA forward pass
def lora_forward(x, W0, A, B, alpha, r):
    # Original computation (frozen)
    base_output = W0 @ x

    # LoRA adaptation (trainable)
    lora_output = (alpha / r) * (B @ (A @ x))

    return base_output + lora_output
</syntaxhighlight>

=== Fused LoRA (Unsloth Optimization) ===

Unsloth fuses the LoRA computation with the base linear operation:

<syntaxhighlight lang="python">
# Standard approach: 3 operations
base = W0 @ x           # O(d*k*n)
down = A @ x            # O(r*k*n)
up = B @ down           # O(d*r*n)
out = base + scale*up   # O(d*n)

# Fused approach: 1 triton kernel
# Computes all operations in a single GPU pass
out = fused_lora_forward(x, W0, A, B, scale)  # Single kernel
</syntaxhighlight>

Benefits:
- Reduced memory bandwidth (single read/write)
- Better GPU utilization
- 2x faster training in practice

=== RSLoRA (Rank-Stabilized LoRA) ===

Standard LoRA uses scaling α/r, but this can cause instability with high ranks. RSLoRA uses:

<math>
\text{scale} = \frac{\alpha}{\sqrt{r}}
</math>

This provides more stable training dynamics when using r > 32.

=== Gradient Checkpointing Integration ===

Unsloth's "smart" gradient checkpointing selectively recomputes:

<syntaxhighlight lang="python">
# Abstract gradient checkpointing strategy
def smart_checkpoint(layer, inputs):
    # Recompute attention (memory-heavy)
    attention = checkpoint(layer.attention, inputs)

    # Cache LoRA gradients (small, fast to recompute)
    lora_grad = checkpoint(layer.lora_forward, inputs)

    # Don't checkpoint final projection (needed for loss)
    output = layer.output_projection(attention + lora_grad)

    return output
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_get_peft_model]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_LoRA_Rank_Selection]]
