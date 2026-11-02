# Principle: DoRA_Weight_Decomposed_Adaptation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|DoRA: Weight-Decomposed Low-Rank Adaptation|https://arxiv.org/abs/2402.09353]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Extension of low-rank adaptation that splits each pretrained weight into a magnitude vector and a direction matrix and applies the low-rank update only to the direction.

=== Description ===

Analysis of full fine-tuning shows it changes magnitude and direction of weight columns in a different pattern than plain low-rank updates do. By training the magnitude as its own parameter vector and the direction through a low-rank delta, the method narrows that gap at a small additional cost.

=== Usage ===

Consider it as a drop-in alternative when low-rank adapters underperform full fine-tuning at the same rank.

== Theoretical Basis ==

The adapted weight is W' = m * (W + B*A) / ||W + B*A||_c, where m is a learned vector, the norm is taken per column, and B*A is a low-rank delta as in the standard method.

== Related Pages ==

* [[implemented_by::Implementation:huggingface_peft_LoraConfig]]
