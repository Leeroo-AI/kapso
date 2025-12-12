{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA Paper|https://arxiv.org/abs/2305.14314]]
* [[source::Repo|Unsloth GitHub|https://github.com/unslothai/unsloth]]
* [[source::Blog|Unsloth LoRA Guide|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Optimization]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Guidelines for selecting which linear layers to apply LoRA adapters to maximize fine-tuning effectiveness while minimizing trainable parameters.

=== Description ===
Not all linear layers in a Transformer contribute equally to fine-tuning performance. Selecting the right `target_modules` determines the trade-off between model capacity, training speed, and memory usage. Unsloth recommends applying LoRA to all linear layers for best results, which their optimizations make feasible.

=== Usage ===
Use this heuristic when configuring `FastLanguageModel.get_peft_model()` to determine which layers to inject LoRA adapters into. Critical decision that affects model quality, training time, and VRAM usage.

== The Insight (Rule of Thumb) ==
* **Action:** Set `target_modules` to include all major linear projection layers.
* **Value (Recommended - All Layers):**
<syntaxhighlight lang="python">
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
    "gate_proj", "up_proj", "down_proj",      # MLP layers
]
</syntaxhighlight>
* **Value (Minimal - Attention Only):**
<syntaxhighlight lang="python">
target_modules = ["q_proj", "v_proj"]  # Minimum viable for basic tasks
</syntaxhighlight>
* **Trade-off:**
  * All layers: Better quality, more VRAM, slower training
  * Attention only: Faster, less VRAM, potentially lower quality

== Layer Selection Guide ==

{| class="wikitable"
! Layer !! Role !! Impact !! Priority
|-
|| q_proj || Query projection || High - critical for attention patterns || Essential
|-
|| k_proj || Key projection || High - attention key computation || Recommended
|-
|| v_proj || Value projection || High - information aggregation || Essential
|-
|| o_proj || Output projection || Medium - final attention output || Recommended
|-
|| gate_proj || MLP gating || Medium - controls information flow || For best quality
|-
|| up_proj || MLP expansion || Medium - feature expansion || For best quality
|-
|| down_proj || MLP compression || Medium - feature compression || For best quality
|}

== Reasoning ==
Research shows that attention layers (Q, K, V, O projections) capture the most task-specific information during fine-tuning. MLP layers (gate, up, down) contribute additional capacity for learning complex patterns. Unsloth's memory optimizations make targeting all layers practical, achieving results competitive with full fine-tuning at a fraction of the cost.

For most use cases, targeting all 7 modules provides the best quality without significant overhead when using Unsloth's optimizations.

== Related Pages ==
=== Used By ===
* [[uses_heuristic::Workflow:QLoRA_Finetuning]]
* [[uses_heuristic::Implementation:Unsloth_get_peft_model]]
* [[uses_heuristic::Principle:Low_Rank_Adaptation]]

