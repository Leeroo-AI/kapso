# Principle: LoRA_Low_Rank_Adaptation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Parameter-efficient method that freezes pretrained weights and learns a low-rank additive update for selected matrices, so only a tiny fraction of parameters is trained.

=== Description ===

Fine-tuning changes a weight matrix by a delta that often has low intrinsic rank. LoRA parametrises that delta as a product of two thin matrices and trains only them. The frozen weights keep the original behaviour available, many deltas can be stored cheaply for one base, and a delta can be merged into the base for inference with no extra latency.

=== Usage ===

Use it when full fine-tuning is too expensive in memory or storage, or when one base must serve several tasks.

== Theoretical Basis ==

For a frozen weight W in R^(d x k), the adapted weight is W' = W + (alpha / r) * B * A, with A in R^(r x k), B in R^(d x r) and rank r much smaller than d and k. A is initialised randomly and B at zero, so training starts exactly at the pretrained function. The trainable parameter count per matrix falls from d*k to r*(d+k).

== Related Pages ==

* [[implemented_by::Implementation:huggingface_peft_LoraConfig]]
* [[uses_heuristic::Heuristic:huggingface_peft_LoRA_Rank_And_Alpha_Selection]]
