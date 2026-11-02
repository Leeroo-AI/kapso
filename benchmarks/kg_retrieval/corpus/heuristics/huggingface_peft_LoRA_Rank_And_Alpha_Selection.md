# Heuristic: LoRA_Rank_And_Alpha_Selection

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Blog|Practical tips for LoRA|https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Guidance for choosing the rank of low-rank adapters and their scaling factor, with common starting values and how both interact with the learning rate.

=== Description ===

Quality saturates quickly with rank, while memory and trainable-parameter count grow linearly. The scaling factor multiplies the learned update, so it acts like a second learning rate on the adapter.

=== Usage ===

Consult it when setting up the first run of an adapter and when results are unstable.

== The Insight (Rule of Thumb) ==

Start with rank 8 to 16 on all linear projections, and a scaling factor equal to the rank or twice the rank. Raise the rank only if training loss plateaus well above that of full fine-tuning. Changing the scaling factor changes the effective step size, so retune the learning rate afterwards.

== Reasoning ==

The update has effective scale alpha / r; keeping that ratio fixed while varying r makes runs comparable.
