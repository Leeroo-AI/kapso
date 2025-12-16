{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|rsLoRA: Rank-Stabilized LoRA|https://arxiv.org/abs/2312.03732]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::PEFT]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Parameter-efficient fine-tuning technique that injects trainable low-rank decomposition matrices into frozen pre-trained model layers, enabling adaptation with a fraction of the parameters.

=== Description ===
Low-Rank Adaptation (LoRA) addresses the challenge of fine-tuning large language models by freezing the pre-trained weights and injecting trainable rank decomposition matrices into each layer. Instead of updating a dense weight matrix W ∈ R^(d×k), LoRA learns two smaller matrices A ∈ R^(r×k) and B ∈ R^(d×r) where r << min(d,k).

The key insight is that the weight updates during fine-tuning have a low intrinsic rank. By constraining updates to a low-rank form, LoRA achieves:
- **Memory efficiency**: Only r × (d + k) parameters per adapted layer vs d × k
- **No inference latency**: Merged weights can be deployed with zero overhead
- **Task switching**: Multiple LoRA adapters can be hot-swapped on a single base model

For transformer models, LoRA is typically applied to:
- **Attention projections**: q_proj, k_proj, v_proj, o_proj
- **MLP projections**: gate_proj, up_proj, down_proj (for SwiGLU architectures)

The rank `r` is the primary hyperparameter controlling capacity vs efficiency tradeoff. Common values are 8, 16, 32, 64, with higher ranks for more complex tasks like reasoning or code generation.

=== Usage ===
Use LoRA configuration when:
- Fine-tuning LLMs on task-specific data (instruction tuning, domain adaptation)
- GPU memory is limited and full fine-tuning is infeasible
- You need to maintain multiple task-specific adapters
- Fast iteration on fine-tuning experiments is required

Rank selection guidelines:
- r=8-16: Simple tasks, conversational fine-tuning
- r=32-64: Moderate complexity, domain-specific knowledge
- r=64-128: Complex reasoning, math, coding tasks (especially for GRPO/RL)

== Theoretical Basis ==
For a pre-trained weight matrix W₀ ∈ R^(d×k), LoRA modifies the forward pass:

<math>
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} BA x
</math>

Where:
- <math>W_0</math>: Frozen pre-trained weights
- <math>B \in \mathbb{R}^{d \times r}</math>: Trainable down-projection (initialized to zeros)
- <math>A \in \mathbb{R}^{r \times k}</math>: Trainable up-projection (initialized from N(0, σ²))
- <math>\alpha</math>: Scaling factor (lora_alpha)
- <math>r</math>: Rank of adaptation

'''Initialization:'''
<syntaxhighlight lang="python">
# Pseudo-code for LoRA initialization
def init_lora_weights(r, d, k):
    # B initialized to zeros ensures ΔW = 0 at start
    B = zeros(d, r)

    # A initialized from normal distribution
    A = randn(r, k) * sqrt(2.0 / r)

    return A, B
</syntaxhighlight>

'''Forward Pass with LoRA:'''
<syntaxhighlight lang="python">
# Pseudo-code for LoRA forward
def lora_forward(x, W0, A, B, alpha, r):
    # Base model computation (frozen)
    base_output = W0 @ x

    # LoRA delta (trainable)
    lora_output = (B @ A @ x) * (alpha / r)

    return base_output + lora_output
</syntaxhighlight>

'''Rank-Stabilized LoRA (rsLoRA):'''
For large ranks, rsLoRA modifies the scaling to maintain stable gradient magnitudes:

<math>
h = W_0 x + \frac{\alpha}{\sqrt{r}} BA x
</math>

This prevents the effective learning rate from scaling with rank, enabling stable training at r=64+ without hyperparameter adjustment.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_get_peft_model]]

=== Tips and Tricks ===
