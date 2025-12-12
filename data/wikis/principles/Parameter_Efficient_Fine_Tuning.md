{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|Prefix Tuning|https://arxiv.org/abs/2101.00190]]
* [[source::Doc|HuggingFace PEFT|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Optimization]], [[domain::Deep_Learning]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Family of techniques that adapt large pre-trained models by training only a small subset of parameters, reducing memory and storage requirements.

=== Description ===
Parameter Efficient Fine-Tuning (PEFT) encompasses methods that achieve model adaptation while updating only a fraction of parameters. This includes LoRA (low-rank adapters), prefix tuning (trainable prompt tokens), adapters (bottleneck layers), and prompt tuning. PEFT methods make fine-tuning of billion-parameter models feasible on consumer hardware and enable efficient multi-task deployment through adapter switching.

=== Usage ===
Use this principle when full fine-tuning is impractical due to memory constraints. Apply when you need to maintain multiple task-specific models efficiently (share base weights, swap adapters). Essential understanding for working with modern LLM fine-tuning where full parameter updates are prohibitively expensive.

== Theoretical Basis ==
'''PEFT Methods Taxonomy:'''

{| class="wikitable"
! Method !! Where !! What !! Params
|-
|| LoRA || Attention/MLP || Low-rank matrices || 0.1-1%
|-
|| Prefix Tuning || Input || Trainable prefixes || <0.1%
|-
|| Adapters || Layers || Bottleneck modules || 1-3%
|-
|| Prompt Tuning || Embeddings || Soft prompts || <0.1%
|-
|| (IA)³ || Activations || Scaling vectors || <0.01%
|}

'''Why PEFT Works:'''
The "intrinsic dimensionality" hypothesis suggests that:
1. Pre-trained models already contain task-relevant features
2. Fine-tuning mainly activates/combines existing capabilities
3. The effective update rank is much lower than full parameter space

'''Mathematical Framework:'''
All PEFT methods can be viewed as constraining the update space:

\[
\theta_{finetuned} = \theta_{pretrained} + \Delta\theta
\]

Where \(\Delta\theta\) is restricted to a low-dimensional subspace:
* LoRA: \(\Delta W = BA\) where \(B, A\) are low-rank
* Prefix: Only modify input representations
* Adapters: Add small trainable modules

'''Comparison:'''
<syntaxhighlight lang="python">
# Full fine-tuning: All 7B parameters trainable
# Memory: ~56GB for optimizer states + gradients

# LoRA fine-tuning: ~40M parameters trainable (0.5%)
# Memory: ~4GB for adapters + optimizer states

# Same final performance on most tasks!
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:Unsloth_get_peft_model]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:LoRA_Rank_Selection]]
* [[uses_heuristic::Heuristic:QLoRA_Target_Modules_Selection]]

