# Principle: QLoRA_Quantized_Low_Rank_Adaptation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Adapter training on top of a base model stored in 4-bit NormalFloat, with double quantization and paged optimizers so very large models fit on one accelerator.

=== Description ===

QLoRA combines three ideas: a 4-bit data type whose levels match the distribution of normally distributed weights, a second round of compression applied to the scaling constants themselves, and optimizer states that spill to host memory under pressure. Gradients flow through the dequantized frozen weights into 16-bit adapters.

=== Usage ===

Choose it when the base model alone does not fit in memory at 16-bit, yet adapter-based tuning of it is still the goal.

== Theoretical Basis ==

Weights are stored in blocks with one scaling constant per block; the constants are themselves compressed to 8 bits. During the forward and backward passes each block is dequantized to the compute dtype just in time, so only the adapters hold trainable, full-precision state.

== Related Pages ==

* [[implemented_by::Implementation:huggingface_peft_LoraConfig]]
* [[implemented_by::Implementation:huggingface_transformers_BitsAndBytes_4bit_Config]]
