{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|RSLoRA|https://arxiv.org/abs/2312.03732]]
* [[source::Paper|DoRA|https://arxiv.org/abs/2402.09353]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Parameter_Efficient]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for configuring Low-Rank Adaptation hyperparameters to balance model capacity, training efficiency, and task performance.

=== Description ===

LoRA Configuration determines how adapter layers modify the base model. The core hyperparameters—rank (r), alpha, target modules, and dropout—control the expressiveness of the adaptation. The configuration represents a trade-off: higher rank increases capacity but also parameter count, while lower rank is more efficient but may underfit complex tasks.

Key configuration decisions include:
* **Rank (r):** The dimensionality of the low-rank matrices (typical: 8-64)
* **Alpha:** Scaling factor that controls the magnitude of updates (alpha/r)
* **Target modules:** Which layers to adapt (attention vs. all linear)
* **Variants:** RSLoRA (rank-stabilized scaling), DoRA (weight decomposition)

=== Usage ===

Apply this principle when designing your LoRA adaptation strategy:
* **Simple tasks:** Use lower rank (r=4-8) with default alpha
* **Complex tasks:** Use higher rank (r=16-64) or DoRA
* **Memory-constrained:** Target only attention (q,v projections)
* **Best performance:** Target all linear layers with moderate rank

== Theoretical Basis ==

'''Low-Rank Decomposition:'''

LoRA approximates weight updates as a low-rank matrix:
<math>\Delta W = BA</math>

Where:
* <math>A \in \mathbb{R}^{r \times k}</math> (down-projection)
* <math>B \in \mathbb{R}^{d \times r}</math> (up-projection)
* <math>r</math> is the rank (hyperparameter)

'''Scaling Factor:'''

The effective update is scaled by alpha/r:
<math>h = W_0 x + \frac{\alpha}{r} \cdot BAx</math>

This scaling ensures that the initialization (B=0) results in the original model behavior.

'''Rank-Stabilized LoRA (RSLoRA):'''

Standard scaling uses <math>\alpha/r</math>, but RSLoRA uses:
<math>h = W_0 x + \frac{\alpha}{\sqrt{r}} \cdot BAx</math>

This stabilizes training at different rank values.

'''DoRA (Weight-Decomposed LoRA):'''

DoRA decomposes weight updates into magnitude and direction:
<math>W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}</math>

Where <math>m</math> is a learnable magnitude vector, improving performance especially at low ranks.

'''Target Module Selection:'''

Pseudo-code for determining which modules to adapt:
<syntaxhighlight lang="python">
# Abstract selection logic
if target_modules == "all-linear":
    # Adapt all nn.Linear layers except output
    targets = find_all_linear_modules(model)
else:
    # Adapt specific modules by name pattern
    targets = match_module_names(model, target_modules)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_LoraConfig_init]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:huggingface_peft_LoRA_Rank_Selection]]
