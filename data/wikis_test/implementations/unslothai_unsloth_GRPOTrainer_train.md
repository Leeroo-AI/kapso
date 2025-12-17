# Implementation: unslothai_unsloth_GRPOTrainer_train

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Wrapper documentation for executing GRPO reinforcement learning training using TRL's GRPOTrainer with Unsloth optimizations.

=== Description ===

GRPOTrainer implements Group Relative Policy Optimization for training language models with reward signals. When Unsloth is imported, it patches GRPOTrainer for:

1. **vLLM-accelerated generation**: Fast sampling during the RL loop
2. **Memory-efficient training**: Optimized gradient computation
3. **Improved logging**: Statistics tracking for reward and policy metrics

GRPO is particularly effective for reasoning tasks where multiple completions can be compared.

=== Usage ===

Use GRPOTrainer after:
1. Loading model with `fast_inference=True`
2. Configuring LoRA with higher rank (64+)
3. Defining reward functions
4. Optionally completing SFT warmup

== Code Reference ==

=== Source Location ===
* '''External Library:''' [https://github.com/huggingface/trl TRL]
* '''Unsloth Patches:''' unsloth/models/rl.py (L500-1349)

=== Signature ===
<syntaxhighlight lang="python">
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model: PreTrainedModel,
    processing_class: PreTrainedTokenizer,
    reward_funcs: List[Callable],
    args: GRPOConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Import unsloth first for patches
import unsloth

from trl import GRPOTrainer, GRPOConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model with vLLM backend and LoRA adapters
|-
| processing_class || PreTrainedTokenizer || Yes || Tokenizer for text processing
|-
| reward_funcs || List[Callable] || Yes || List of reward functions
|-
| args || GRPOConfig || Yes || GRPO training configuration
|-
| train_dataset || Dataset || Yes || Dataset with "prompt" column
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| TrainOutput || namedtuple || Contains global_step, training metrics
|}

== Usage Examples ==

=== Complete GRPO Training Setup ===
<syntaxhighlight lang="python">
import unsloth
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. Load model with vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.6,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(
    model, r = 64, lora_alpha = 64,
)

# 3. Define reward function
def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer):
        if str(ans) in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 4. Configure GRPO
grpo_config = GRPOConfig(
    output_dir = "grpo_outputs",
    learning_rate = 5e-6,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 1,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 1,
    logging_steps = 1,
    report_to = "none",
)

# 5. Load dataset
dataset = load_dataset("your_dataset", split="train")

# 6. Create trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [correctness_reward],
    args = grpo_config,
    train_dataset = dataset,
)

# 7. Train!
trainer.train()
</syntaxhighlight>

=== Multiple Reward Functions ===
<syntaxhighlight lang="python">
# Define multiple reward signals
def format_reward(completions, **kwargs):
    return [0.5 if "\\boxed{" in c else 0.0 for c in completions]

def length_reward(completions, **kwargs):
    return [min(len(c) / 500, 1.0) for c in completions]

def correctness_reward(completions, answer, **kwargs):
    # ... as above

# Combine in trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        format_reward,
        length_reward,
        correctness_reward,
    ],  # All rewards are combined
    args = grpo_config,
    train_dataset = dataset,
)
</syntaxhighlight>

=== With SFT Warmup ===
<syntaxhighlight lang="python">
from trl import SFTTrainer, SFTConfig

# Phase 1: SFT warmup
sft_config = SFTConfig(
    output_dir = "sft_outputs",
    max_steps = 50,
    learning_rate = 2e-4,
)

sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = sft_dataset,
    args = sft_config,
)
sft_trainer.train()

# Phase 2: GRPO training
grpo_trainer = GRPOTrainer(
    model = model,  # Continue from SFT
    processing_class = tokenizer,
    reward_funcs = [reward_func],
    args = grpo_config,
    train_dataset = rl_dataset,
)
grpo_trainer.train()
</syntaxhighlight>

== GRPOConfig Key Parameters ==

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_generations` | 8 | Completions sampled per prompt |
| `max_prompt_length` | 256 | Maximum prompt token length |
| `max_completion_length` | 256 | Maximum completion token length |
| `learning_rate` | 5e-6 | Lower than SFT (RL is noisier) |
| `kl_coef` | 0.1 | KL divergence penalty coefficient |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_GRPO_Training]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
