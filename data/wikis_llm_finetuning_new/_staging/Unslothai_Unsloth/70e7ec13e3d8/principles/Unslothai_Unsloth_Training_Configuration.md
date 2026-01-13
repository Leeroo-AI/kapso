# Principle: Training_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Trainer Documentation|https://huggingface.co/docs/transformers/trainer]]
* [[source::Doc|TRL SFTTrainer Documentation|https://huggingface.co/docs/trl/sft_trainer]]
* [[source::Paper|QLoRA|https://arxiv.org/abs/2305.14314]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Training]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for configuring training hyperparameters and optimization settings for supervised fine-tuning of language models.

=== Description ===

Training Configuration defines the hyperparameters that control the fine-tuning process. Key settings include:

* **Learning Rate**: How quickly the model adapts (typically 1e-4 to 5e-5 for LoRA)
* **Batch Size**: Number of samples processed together (affects memory and convergence)
* **Gradient Accumulation**: Simulates larger batches by accumulating gradients across steps
* **Epochs/Steps**: Total training duration
* **Warmup**: Gradual learning rate increase at start to stabilize training
* **Weight Decay**: Regularization to prevent overfitting

These parameters interact with each other and the model architecture to determine training efficiency, convergence speed, and final model quality.

=== Usage ===

Use this principle when:
* Setting up a training run after preparing model and data
* Tuning hyperparameters for better performance
* Balancing training speed against memory constraints
* Configuring checkpointing and logging

This step comes after data formatting and before starting the training loop.

== Theoretical Basis ==

The key hyperparameter relationships:

'''Effective Batch Size:'''
<math>
\text{effective\_batch\_size} = \text{per\_device\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{num\_gpus}
</math>

'''Learning Rate Scaling:'''
- Larger effective batch sizes often benefit from higher learning rates
- The "linear scaling rule": scale LR proportionally to batch size

'''Warmup Strategy:'''
<syntaxhighlight lang="python">
# Linear warmup over warmup_steps
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    lr = scheduler(step)  # Cosine, linear decay, etc.
</syntaxhighlight>

'''Key Hyperparameter Guidelines for QLoRA:'''
{| class="wikitable"
|-
! Parameter !! Typical Range !! Notes
|-
| learning_rate || 1e-4 to 5e-5 || Higher OK for LoRA than full fine-tuning
|-
| per_device_train_batch_size || 1-8 || Constrained by GPU memory
|-
| gradient_accumulation_steps || 2-16 || Increase if batch size limited
|-
| num_train_epochs || 1-5 || More for small datasets
|-
| warmup_ratio || 0.03-0.1 || Fraction of total steps
|-
| weight_decay || 0.01-0.1 || Regularization strength
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_SFTConfig]]

