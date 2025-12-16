{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Fine_Tuning]], [[domain::GRPO]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Concrete tool for patching TRL (Transformer Reinforcement Learning) trainers with Unsloth optimizations for memory-efficient RL training.

=== Description ===

`PatchFastRL` is a function that dynamically patches TRL's reinforcement learning trainers to enable:

* **80% VRAM reduction**: Optimized gradient computation and caching
* **vLLM integration**: Fast batch generation during RL sampling phases
* **Automatic trainer patching**: Patches GRPO, PPO, DPO, ORPO, KTO trainers
* **Statistics logging**: Improved metrics tracking for RL training

The function works by:
1. Patching TRL trainer classes to use Unsloth's optimized functions
2. Setting up model state switching between inference and training modes
3. Configuring vLLM for fast generation when `fast_inference=True`
4. Replacing memory-intensive operations with efficient alternatives

=== Usage ===

Call `PatchFastRL()` before importing TRL trainers to enable optimizations. This must be done at the start of any RL training script.

Use cases:
* GRPO training for reasoning enhancement
* PPO training with reward models
* Preference-based training (DPO, ORPO, KTO)
* Any TRL-based RL workflow

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unslothai_unsloth]
* '''File:''' [https://github.com/unslothai/unsloth/blob/main/unsloth/models/rl.py unsloth/models/rl.py]
* '''Lines:''' 1343-1349 (PatchFastRL), 61-200 (PatchRL)

=== Signature ===
<syntaxhighlight lang="python">
def PatchFastRL(
    algorithm: Optional[str] = None,
    FastLanguageModel = None,
):
    """
    Patch TRL trainers with Unsloth optimizations.

    Args:
        algorithm: RL algorithm name for metrics ("grpo", "ppo", "dpo", etc.)
                  If None, patches all trainers without specific metrics.
        FastLanguageModel: The FastLanguageModel class to use for model
                          state switching (inference/training modes).

    Usage:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("grpo", FastLanguageModel)

        # Now import TRL trainers - they're automatically optimized
        from trl import GRPOTrainer, GRPOConfig
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, PatchFastRL
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| algorithm || str || No || RL algorithm name for metrics: "grpo", "ppo", "dpo", "orpo", "kto"
|-
| FastLanguageModel || class || No || FastLanguageModel class for inference/training mode switching
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| None || None || Function patches TRL in-place, no return value
|}

== Usage Examples ==

=== Basic GRPO Setup ===
<syntaxhighlight lang="python">
# IMPORTANT: Import and patch BEFORE importing TRL trainers
from unsloth import FastLanguageModel, PatchFastRL

# Patch TRL with Unsloth optimizations
PatchFastRL("grpo", FastLanguageModel)

# Now import TRL - trainers are automatically patched
import torch
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

# Load model with vLLM for fast generation
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
    fast_inference=True,        # Enable vLLM
    gpu_memory_utilization=0.6, # Reserve for gradients
    max_lora_rank=64,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
)

# Define reward function
def reward_fn(completions, prompts):
    rewards = []
    for completion in completions:
        # Your reward logic here
        reward = len(completion.split()) * 0.01  # Example
        rewards.append(reward)
    return rewards

# Configure and train
config = GRPOConfig(
    output_dir="grpo_output",
    per_device_train_batch_size=1,
    num_generations=8,
    max_new_tokens=1024,
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    reward_funcs=reward_fn,
)

trainer.train()
</syntaxhighlight>

=== PPO Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, PatchFastRL

# Patch for PPO
PatchFastRL("ppo", FastLanguageModel)

from trl import PPOConfig, PPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(model, r=32)

config = PPOConfig(
    output_dir="ppo_output",
    learning_rate=1e-5,
)

trainer = PPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
)
</syntaxhighlight>

=== DPO Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, PatchFastRL

# Patch for DPO (preference-based)
PatchFastRL("dpo", FastLanguageModel)

from trl import DPOConfig, DPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(model, r=16)

# DPO requires paired preference data
# {"prompt": "...", "chosen": "...", "rejected": "..."}
config = DPOConfig(
    output_dir="dpo_output",
    beta=0.1,  # KL penalty
)

trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
</syntaxhighlight>

=== Inference Mode Switching ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel, PatchFastRL

PatchFastRL("grpo", FastLanguageModel)

model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, r=64)

# During training, Unsloth automatically handles mode switching:
# - Generation phase: FastLanguageModel.for_inference(model)
# - Optimization phase: FastLanguageModel.for_training(model)

# Manual mode switching if needed:
FastLanguageModel.for_inference(model)
outputs = model.generate(...)

FastLanguageModel.for_training(model)
loss = model(**batch).loss
</syntaxhighlight>

== Supported Trainers ==

{| class="wikitable"
|-
! Trainer !! Algorithm !! Use Case
|-
| GRPOTrainer || grpo || Reasoning, math, coding enhancement
|-
| PPOTrainer || ppo || General RL with reward model
|-
| DPOTrainer || dpo || Direct preference optimization
|-
| ORPOTrainer || orpo || Odds ratio preference optimization
|-
| KTOTrainer || kto || Kahneman-Tversky optimization
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_RL_Setup]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_vLLM]]
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_RL_Hyperparameters]]
* [[uses_heuristic::Heuristic:unslothai_unsloth_LoRA_Rank_Selection]]
