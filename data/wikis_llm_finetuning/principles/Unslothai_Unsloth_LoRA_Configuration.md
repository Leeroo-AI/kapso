# Principle: LoRA_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Paper|RSLoRA|https://arxiv.org/abs/2312.03732]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Parameter_Efficient_Finetuning]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for adding trainable low-rank decomposition matrices to frozen pre-trained weights, enabling parameter-efficient fine-tuning with minimal memory overhead.

=== Description ===

Low-Rank Adaptation (LoRA) injects trainable rank decomposition matrices into transformer layers while keeping pre-trained weights frozen. Instead of updating a weight matrix W ∈ R^(d×k), LoRA adds a parallel path through two smaller matrices: A ∈ R^(r×k) and B ∈ R^(d×r), where r << min(d, k).

This approach reduces trainable parameters by orders of magnitude while achieving comparable performance to full fine-tuning. For a 7B parameter model, typical LoRA configurations train only 0.1-1% of total parameters.

The principle extends to QLoRA by applying LoRA to quantized base models, combining the memory savings of quantization with the efficiency of low-rank adaptation.

=== Usage ===

Apply LoRA configuration when:
* Fine-tuning large models with limited GPU memory
* Training task-specific adapters for deployment
* Rapid experimentation with different fine-tuning objectives
* Multi-task learning (separate adapters per task)

Key configuration decisions:
* **Rank (r)**: Higher rank = more capacity but more memory/compute
* **Alpha**: Scaling factor; alpha/r determines effective learning rate multiplier
* **Target modules**: Which layers receive LoRA adapters

== Theoretical Basis ==

=== Low-Rank Decomposition ===

For a pre-trained weight matrix W₀, LoRA parameterizes the update as:

<math>
W = W_0 + \Delta W = W_0 + BA
</math>

Where:
* W₀ ∈ R^(d×k) is frozen
* B ∈ R^(d×r) initialized to zeros
* A ∈ R^(r×k) initialized from N(0, σ²)
* r << min(d, k) is the rank

=== Forward Pass ===

<math>
h = W_0 x + \frac{\alpha}{r} BA x
</math>

The scaling factor α/r controls the magnitude of the LoRA contribution relative to the base weights.

=== Parameter Efficiency ===

For a single linear layer:
* Full fine-tuning: d × k parameters
* LoRA: r × (d + k) parameters

With r=16, d=4096, k=4096:
* Full: 16.7M parameters
* LoRA: 131K parameters (0.8% of full)

=== Rank-Stabilized LoRA (RSLoRA) ===

Standard LoRA scales by α/r. RSLoRA instead uses:

<math>
h = W_0 x + \frac{\alpha}{\sqrt{r}} BA x
</math>

This stabilizes training across different rank values.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# LoRA forward pass (abstract)
class LoRALinear:
    def __init__(self, base_layer, r, alpha):
        self.base = base_layer  # Frozen W0
        self.lora_A = init_kaiming(r, in_features)  # Trainable
        self.lora_B = init_zeros(out_features, r)   # Trainable
        self.scale = alpha / r  # or alpha / sqrt(r) for RSLoRA

    def forward(self, x):
        base_out = self.base(x)  # W0 @ x (dequantized if 4-bit)
        lora_out = (self.lora_B @ self.lora_A @ x) * self.scale
        return base_out + lora_out
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_peft_model]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection_Tip]]
