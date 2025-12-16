{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::LoRA]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of applying LoRA adapters with higher rank configuration for reinforcement learning capacity.

=== Description ===

RL LoRA setup differs from SFT:
- Higher rank (r=32-128) for capturing complex reasoning patterns
- All attention and MLP layers typically targeted
- Gradient checkpointing essential for memory

== Practical Guide ==

=== High-Rank LoRA for RL ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model,
    r=64,              # Higher rank for RL
    lora_alpha=64,     # Match alpha to rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

=== Rank Selection Guidelines ===
<syntaxhighlight lang="python">
# r=32: Basic reasoning tasks
# r=64: Complex reasoning, math (recommended)
# r=128: Maximum capacity, high memory

# Rule of thumb: RL benefits from higher ranks
# than SFT because policy gradients require
# more model capacity to capture subtle patterns
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_PatchFastRL]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
