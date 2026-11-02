# Workflow: QLoRA_Finetune_7B_On_Instruction_Data

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|QLoRA|https://github.com/artidoro/qlora]]
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::LLMs]], [[domain::Fine_Tuning]], [[domain::PEFT]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

Recipe for teaching a seven-billion-parameter language model to follow instructions by training low-rank adapters over a frozen base held in 4-bit precision.

=== Description ===

The base model stays frozen and is stored in a compressed 4-bit format; small trainable adapter matrices are attached to its linear layers and are the only parameters that receive gradients. This keeps the memory of a 7B model within reach of a single mid-range accelerator.

=== Usage ===

Follow it to specialise a pretrained chat or base model on a set of prompt-and-response examples without full fine-tuning.

== Execution Steps ==

=== Step 1: Format the examples ===

Render every example into one text sequence using the model family's chat template, so prompt and response boundaries are explicit and the loss can be limited to the response.

=== Step 2: Load the compressed base ===

Open the pretrained checkpoint with its weights stored in 4-bit NormalFloat, with a higher-precision compute type for the matrix products.

=== Step 3: Attach adapters ===

Insert low-rank adapter matrices into the attention and feed-forward projections and mark only those as trainable.

=== Step 4: Train ===

Optimise the adapters with a paged optimizer, a modest learning rate and a cosine schedule, evaluating on held-out examples between epochs.

=== Step 5: Save ===

Store the adapter weights separately from the base, or fold them into a full-precision copy of the base for deployment.

== GitHub URL ==

[https://github.com/artidoro/qlora artidoro/qlora]
