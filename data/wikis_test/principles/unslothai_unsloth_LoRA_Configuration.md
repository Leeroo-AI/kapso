# Principle: unslothai_unsloth_LoRA_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|DoRA: Weight-Decomposed Low-Rank Adaptation|https://arxiv.org/abs/2402.09353]]
|-
! Domains
| [[domain::NLP]], [[domain::Parameter_Efficient]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for configuring Low-Rank Adaptation (LoRA) adapters to enable parameter-efficient fine-tuning of large language models.

=== Description ===

LoRA Configuration determines how low-rank matrices are injected into a frozen pre-trained model to create trainable adapters. Instead of fine-tuning all parameters, LoRA:

1. **Freezes the pre-trained weights** W
2. **Adds trainable low-rank matrices** A and B where the adapted weight is W' = W + BA
3. **Reduces trainable parameters** from millions to thousands while maintaining performance

Key configuration choices affect the balance between model capacity, memory usage, and training stability.

=== Usage ===

Use this principle when:
- Setting up parameter-efficient fine-tuning for any QLoRA or standard LoRA workflow
- Deciding which layers to adapt and with what capacity
- Balancing memory constraints against model expressiveness

This is the **second trainable step** in QLoRA workflows, applied immediately after model loading.

== Theoretical Basis ==

=== Low-Rank Decomposition ===

For a pre-trained weight matrix <math>W \in \mathbb{R}^{d \times k}</math>:

<math>
W' = W + \Delta W = W + BA
</math>

Where:
* <math>B \in \mathbb{R}^{d \times r}</math> (down-projection)
* <math>A \in \mathbb{R}^{r \times k}</math> (up-projection)
* <math>r \ll \min(d, k)</math> is the rank

'''Parameter Reduction:'''
<syntaxhighlight lang="python">
# Full fine-tuning parameters
full_params = d * k

# LoRA parameters
lora_params = d * r + r * k = r * (d + k)

# For d=k=4096, r=16:
# Full: 16,777,216 params
# LoRA: 131,072 params (0.78%)
</syntaxhighlight>

=== Scaling Factor ===

The LoRA output is scaled by <math>\alpha / r</math>:

<math>
h = Wx + \frac{\alpha}{r}BAx
</math>

This scaling ensures stable training regardless of rank choice:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Effective learning rate for LoRA weights
# With alpha = r, effective_lr = base_lr (standard scaling)
# With alpha = 2*r, effective_lr = 2 * base_lr (aggressive)

def lora_forward(x, W, A, B, alpha, r):
    # Frozen pre-trained output
    base_output = W @ x

    # Trainable LoRA output with scaling
    lora_output = (alpha / r) * (B @ A @ x)

    return base_output + lora_output
</syntaxhighlight>

=== Rank-Stabilized LoRA (RSLoRA) ===

RSLoRA uses different scaling to improve training stability at high ranks:

<math>
h = Wx + \frac{\alpha}{\sqrt{r}}BAx
</math>

<syntaxhighlight lang="python">
# Standard LoRA scaling (default)
scale = alpha / r

# RSLoRA scaling (use_rslora=True)
scale = alpha / sqrt(r)

# RSLoRA provides more stable gradients for r > 32
</syntaxhighlight>

=== Target Module Selection ===

Different modules have different impacts:

{| class="wikitable"
|-
! Module Type !! Typical Modules !! Impact
|-
| Attention Query/Key/Value || q_proj, k_proj, v_proj || Controls what the model attends to
|-
| Attention Output || o_proj || Affects attention aggregation
|-
| MLP Gate/Up || gate_proj, up_proj || Controls feature activation
|-
| MLP Down || down_proj || Projects features back to hidden dimension
|}

'''Pseudo-code for target selection:'''
<syntaxhighlight lang="python">
# Default: all attention + MLP (most expressive)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

# Attention-only (faster, less capacity)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# QKV-only (minimal, sometimes sufficient)
target_modules = ["q_proj", "k_proj", "v_proj"]
</syntaxhighlight>

=== Initialization ===

LoRA matrices are initialized to produce zero output initially:

<syntaxhighlight lang="python">
# A: Random initialization (small values)
A = random.normal(0, 1/sqrt(r), size=(r, k))

# B: Zero initialization
B = zeros(d, r)

# Initial output: BA = 0, so W' = W (starts from pre-trained)
</syntaxhighlight>

== Practical Guide ==

=== Rank Selection Guidelines ===

| Task Complexity | Recommended Rank | Notes |
|-----------------|------------------|-------|
| Simple classification | 8-16 | Minimal capacity needed |
| Instruction following | 16-32 | Standard choice |
| Complex reasoning | 32-64 | More capacity for multi-step |
| Math/Code | 64-128 | Highest capacity |

=== Memory vs Capacity Tradeoff ===

<syntaxhighlight lang="python">
# Memory estimate for LoRA parameters (fp16)
def estimate_lora_memory(model_hidden_dim, rank, num_layers):
    # Per layer: 2 matrices per target module
    # Typical: 7 target modules per layer
    params_per_layer = 2 * rank * model_hidden_dim * 7
    total_params = params_per_layer * num_layers

    # Memory in GB (fp16)
    memory_gb = total_params * 2 / (1024**3)
    return memory_gb

# Example: 7B model (32 layers, 4096 hidden)
# Rank 16: ~0.03 GB
# Rank 64: ~0.12 GB
# Rank 256: ~0.48 GB
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_get_peft_model]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_QLoRA_Finetuning]]
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
