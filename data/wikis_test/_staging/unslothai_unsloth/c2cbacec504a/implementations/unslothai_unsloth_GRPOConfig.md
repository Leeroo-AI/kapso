# Implementation: unslothai_unsloth_GRPOConfig

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|TRL GRPOConfig|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Wrapper documentation for configuring GRPO training hyperparameters using TRL's GRPOConfig.

=== Description ===

GRPOConfig defines all hyperparameters for GRPO training including:
- Generation settings (num_generations, max tokens)
- Learning rate and optimization
- KL divergence penalty
- Logging and checkpointing

=== Usage ===

Use GRPOConfig when setting up GRPOTrainer to specify training behavior.

== Code Reference ==

=== Source Location ===
* '''External Library:''' TRL
* '''Unsloth Patches:''' unsloth/models/rl.py (L1-500)

=== Signature ===
<syntaxhighlight lang="python">
from trl import GRPOConfig

config = GRPOConfig(
    output_dir: str = "grpo_outputs",
    learning_rate: float = 5e-6,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    num_generations: int = 8,
    max_prompt_length: int = 256,
    max_completion_length: int = 256,
    num_train_epochs: int = 1,
    kl_coef: float = 0.1,
    logging_steps: int = 1,
    save_steps: int = 100,
    report_to: str = "none",
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from trl import GRPOConfig
</syntaxhighlight>

== Key Parameters ==

{| class="wikitable"
|-
! Parameter !! Default !! Description
|-
| num_generations || 8 || Completions per prompt
|-
| max_completion_length || 256 || Max tokens to generate
|-
| learning_rate || 5e-6 || Lower than SFT
|-
| kl_coef || 0.1 || KL divergence penalty
|}

== Usage Example ==

<syntaxhighlight lang="python">
from trl import GRPOConfig

config = GRPOConfig(
    output_dir = "grpo_outputs",
    learning_rate = 5e-6,
    per_device_train_batch_size = 4,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 1,
    kl_coef = 0.1,
    logging_steps = 1,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_GRPO_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
