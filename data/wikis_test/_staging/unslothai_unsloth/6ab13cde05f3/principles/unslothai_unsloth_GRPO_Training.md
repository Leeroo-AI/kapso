{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Repo|TRL|https://github.com/huggingface/trl]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of running the GRPO training loop to optimize model policy based on reward signals.

=== Description ===

GRPO training loop:
1. Sample multiple completions per prompt
2. Compute rewards for each completion
3. Calculate group-relative advantages
4. Update policy to increase probability of high-reward outputs

== Practical Guide ==

=== Initialize and Run Trainer ===
<syntaxhighlight lang="python">
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

# Run training
trainer.train()
</syntaxhighlight>

=== Monitor Training ===
<syntaxhighlight lang="python">
# Key metrics to watch:
# - reward/mean: Should increase over time
# - reward/std: Lower is better (more consistent)
# - kl: Should stay reasonable (< 0.1)
# - loss: General optimization health

# With W&B
trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(
        output_dir="outputs",
        report_to="wandb",
        run_name="grpo-experiment",
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)
</syntaxhighlight>

=== Resume Training ===
<syntaxhighlight lang="python">
# From checkpoint
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
    resume_from_checkpoint="outputs/checkpoint-500",
)

trainer.train()
</syntaxhighlight>

=== Early Stopping ===
<syntaxhighlight lang="python">
from transformers import EarlyStoppingCallback

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)
    ],
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_PatchFastRL]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
