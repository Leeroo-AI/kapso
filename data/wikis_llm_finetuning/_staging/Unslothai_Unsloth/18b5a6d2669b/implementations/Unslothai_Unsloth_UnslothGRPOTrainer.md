# Implementation: UnslothGRPOTrainer

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Training]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Wrapper for TRL's GRPOTrainer that integrates Unsloth's vLLM optimization and efficient gradient computation for GRPO reinforcement learning.

=== Description ===

`UnslothGRPOTrainer` extends TRL's `GRPOTrainer` with:
* **vLLM integration** - Uses model's attached vLLM engine for fast generation
* **Inference mode switching** - Automatic for_inference/for_training mode management
* **Memory optimization** - Chunked gradient computation via `unsloth_num_chunks`

This is a **Wrapper Doc** - it documents how Unsloth extends TRL's trainer.

=== Usage ===

Create `UnslothGRPOTrainer` with the vLLM-enabled model, tokenizer, reward functions, and config. Call `trainer.train()` to execute GRPO training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/rl.py
* '''Lines:''' L240-700 (dynamically generated)

=== External Reference ===
* '''Library:''' [https://github.com/huggingface/trl TRL]
* '''Base Class:''' `trl.GRPOTrainer`

=== Signature ===
<syntaxhighlight lang="python">
class UnslothGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer with Unsloth optimizations.
    """
    def __init__(
        self,
        model: PreTrainedModel,  # Model with vLLM engine
        processing_class: PreTrainedTokenizer,  # Tokenizer
        reward_funcs: List[Callable],  # List of reward functions
        args: UnslothGRPOConfig = None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        **kwargs,
    ):
        """
        Initialize GRPO trainer.

        Args:
            model: Model with vLLM from from_pretrained(fast_inference=True)
            processing_class: Tokenizer (named processing_class in TRL >= 0.13)
            reward_funcs: List of reward functions [f(completions, prompts) -> scores]
            args: UnslothGRPOConfig with training hyperparameters
            train_dataset: Dataset with "prompt" column

        Note: TRL >= 0.13 renamed tokenizer to processing_class
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import UnslothGRPOTrainer, UnslothGRPOConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Model with vLLM engine and LoRA adapters
|-
| processing_class || PreTrainedTokenizer || Yes || Tokenizer with chat template
|-
| reward_funcs || List[Callable] || Yes || Reward functions (scores summed)
|-
| args || UnslothGRPOConfig || No || Training configuration
|-
| train_dataset || Dataset || Yes || Dataset with "prompt" column
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| trainer || UnslothGRPOTrainer || Configured trainer
|-
| train() returns || TrainOutput || Training metrics (rewards, loss, etc.)
|}

== Usage Examples ==

=== Complete GRPO Training ===
<syntaxhighlight lang="python">
from unsloth import (
    FastLanguageModel,
    UnslothGRPOTrainer,
    UnslothGRPOConfig,
)
from unsloth.chat_templates import get_chat_template
import re

# 1. Load model with vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.5,
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(model, r=64)

# 3. Configure chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# 4. Define reward functions
def format_reward(completions, prompts, **kwargs):
    rewards = []
    for c in completions:
        score = 0.5 if "<think>" in c else 0.0
        score += 0.5 if "\\boxed{" in c else 0.0
        rewards.append(score)
    return rewards

def correctness_reward(completions, prompts, **kwargs):
    # Placeholder - implement actual verification
    return [0.0] * len(completions)

# 5. Configure GRPO
grpo_config = UnslothGRPOConfig(
    output_dir = "./grpo_outputs",
    num_generations = 6,
    max_completion_length = 200,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    beta = 0.1,
    learning_rate = 5e-6,
    max_steps = 100,
    logging_steps = 1,
    bf16 = True,
)

# 6. Create trainer
trainer = UnslothGRPOTrainer(
    model = model,
    processing_class = tokenizer,  # Use processing_class for TRL >= 0.13
    reward_funcs = [format_reward, correctness_reward],
    args = grpo_config,
    train_dataset = dataset,  # Must have "prompt" column
)

# 7. Train
trainer.train()

# 8. Save model
model.save_pretrained_merged("./grpo_model", tokenizer, save_method="merged_16bit")
</syntaxhighlight>

=== Math Reasoning Training ===
<syntaxhighlight lang="python">
from unsloth import UnslothGRPOTrainer, UnslothGRPOConfig

# Assume model and tokenizer already loaded with vLLM

# Math-specific reward functions
def format_reward(completions, prompts, **kwargs):
    """Check for proper reasoning format."""
    rewards = []
    for c in completions:
        score = 0.0
        if "<think>" in c and "</think>" in c:
            score += 0.3
        if "\\boxed{" in c:
            score += 0.2
        rewards.append(score)
    return rewards

def correctness_reward(completions, prompts, answers, **kwargs):
    """Check if answer is correct."""
    rewards = []
    for completion, answer in zip(completions, answers):
        match = re.search(r"\\boxed\{(.+?)\}", completion)
        if match and match.group(1).strip() == str(answer).strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

grpo_config = UnslothGRPOConfig(
    output_dir = "./math_grpo",
    num_generations = 16,  # More generations for math
    max_completion_length = 500,  # Longer for reasoning
    beta = 0.05,
    learning_rate = 1e-5,
    max_steps = 500,
    bf16 = True,
)

trainer = UnslothGRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [format_reward, correctness_reward],
    args = grpo_config,
    train_dataset = math_dataset,
)

trainer.train()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_GRPO_Execution]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection_Tip]]
