{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Doc|Transformers Trainer|https://huggingface.co/docs/transformers/main_classes/trainer]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Quantization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Principle for executing training on QLoRA models with appropriate settings for quantized training stability and memory efficiency.

=== Description ===

QLoRA Training Execution runs the training loop on a quantized PEFT model. Key considerations:

* **Gradient accumulation**: Often required to achieve effective batch sizes within memory constraints
* **Paged optimizers**: Use `paged_adamw_8bit` for additional memory savings
* **Gradient checkpointing**: Trade compute for memory when needed
* **Precision handling**: Monitor for NaN losses indicating precision issues

=== Usage ===

Apply when training a QLoRA model:
* Use gradient accumulation to simulate larger batches
* Enable gradient checkpointing if memory is still tight
* Choose appropriate paged optimizer
* Monitor loss for stability

== Theoretical Basis ==

'''Memory-Compute Tradeoffs:'''

| Technique | Memory Saved | Compute Cost |
|-----------|--------------|--------------|
| 4-bit quantization | ~4x | ~10-20% slower |
| Gradient checkpointing | ~50% | ~20% slower |
| Paged optimizer | ~30% | Minimal |
| Gradient accumulation | N/A | N/A (same math) |

'''Effective Batch Size:'''

<math>\text{effective\_batch} = \text{per\_device\_batch} \times \text{grad\_accum\_steps} \times \text{num\_devices}</math>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:huggingface_peft_Trainer_train_qlora]]
