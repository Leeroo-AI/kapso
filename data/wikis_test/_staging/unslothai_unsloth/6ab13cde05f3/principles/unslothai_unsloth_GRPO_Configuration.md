{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Repo|TRL|https://github.com/huggingface/trl]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::GRPO]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of configuring GRPO (Group Relative Policy Optimization) training hyperparameters for optimal reinforcement learning.

=== Description ===

GRPO configuration controls:
- Generation settings (samples per prompt, temperature)
- Policy optimization (beta, loss type)
- Training dynamics (learning rate, batch size)

== Practical Guide ==

=== Basic GRPO Config ===
<syntaxhighlight lang="python">
from trl import GRPOConfig
import torch

config = GRPOConfig(
    # Output
    output_dir="grpo_outputs",

    # Batch settings
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # GRPO specific
    num_generations=8,          # Samples per prompt
    max_new_tokens=1024,        # Max generation length
    temperature=0.7,            # Sampling temperature

    # Policy optimization
    beta=0.001,                 # KL penalty
    loss_type="bnpo",           # GRPO variant

    # Training
    learning_rate=5e-6,         # Lower than SFT
    max_steps=500,
    logging_steps=1,
    save_steps=100,

    # Optimizer
    optim="adamw_8bit",
    warmup_ratio=0.1,

    # Precision
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
)
</syntaxhighlight>

=== Key Hyperparameters ===
<syntaxhighlight lang="python">
# num_generations: More = better advantage estimation, slower
# 4: Fast training
# 8: Balanced (recommended)
# 16: Better gradients, 2x slower

# beta: KL penalty strength
# 0.0001-0.001: Weak constraint, more exploration
# 0.01: Moderate constraint
# 0.1: Strong constraint, stable but limited learning

# temperature: Generation diversity
# 0.5: More deterministic
# 0.7: Balanced (recommended)
# 1.0: More diverse

# loss_type options:
# "grpo": Standard GRPO
# "bnpo": Bounded normalized (recommended)
# "dr_grpo": Dr. GRPO with reward scaling
# "dapo": Dynamic advantage
</syntaxhighlight>

=== Aggressive Learning Config ===
<syntaxhighlight lang="python">
# For faster learning (may be less stable)
config = GRPOConfig(
    num_generations=4,
    beta=0.0001,
    learning_rate=1e-5,
    temperature=0.8,
)
</syntaxhighlight>

=== Conservative Config ===
<syntaxhighlight lang="python">
# For stable learning
config = GRPOConfig(
    num_generations=16,
    beta=0.01,
    learning_rate=2e-6,
    temperature=0.6,
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_PatchFastRL]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
