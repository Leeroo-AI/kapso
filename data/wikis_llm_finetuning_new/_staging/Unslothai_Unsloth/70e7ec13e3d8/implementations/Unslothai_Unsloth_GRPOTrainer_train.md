# Implementation: GRPOTrainer_train

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Repo|TRL|https://github.com/huggingface/trl]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for GRPO reinforcement learning training provided by TRL and optimized by Unsloth.

=== Description ===

`GRPOTrainer` (from TRL) implements the GRPO algorithm for training language models with reward-based feedback. Unsloth patches it via `PatchFastRL` to provide:

* vLLM integration for fast generation during training
* Optimized memory management
* LoRA adapter handling with vLLM
* Efficient gradient computation

The trainer manages the full GRPO loop: generation, reward computation, advantage calculation, and policy updates.

=== Usage ===

Import after calling `PatchFastRL` (or simply import from unsloth first). Create a GRPOTrainer with your model, config, dataset, and reward functions, then call `.train()` to start RL training.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/rl.py
* '''Lines:''' 1437-1444 (PatchFastRL), 388-1127 (_patch_trl_rl_trainers)

=== Signature ===
<syntaxhighlight lang="python">
# From TRL (patched by Unsloth)
class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[PreTrainedTokenizer] = None,
        reward_funcs: Optional[List[Callable]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        **kwargs,
    ):
        """
        GRPO Trainer for reinforcement learning.

        Args:
            model: Model with LoRA adapters and vLLM engine
            args: GRPOConfig with RL hyperparameters
            train_dataset: Dataset with prompts
            processing_class: Tokenizer
            reward_funcs: List of reward functions
            callbacks: Training callbacks
        """

    def train(self) -> TrainOutput:
        """Execute GRPO training loop."""


class GRPOConfig(TrainingArguments):
    def __init__(
        self,
        # RL-specific parameters
        num_generations: int = 4,
        max_completion_length: int = 256,
        beta: float = 0.001,
        loss_type: str = "grpo",
        temperature: float = 0.9,
        use_vllm: bool = True,
        # ... standard training args ...
    ):
        """
        Configuration for GRPO training.

        Args:
            num_generations: Completions per prompt for advantage estimation
            max_completion_length: Maximum tokens per completion
            beta: KL penalty coefficient (lower = more change)
            loss_type: "grpo", "bnpo", "dr_grpo", "dapo"
            temperature: Sampling temperature for generation
            use_vllm: Use vLLM for generation (auto-True with fast_inference)
        """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel  # Import first to apply patches
from trl import GRPOTrainer, GRPOConfig
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PeftModelForCausalLM || Yes || Model with LoRA adapters and vllm_engine
|-
| args || GRPOConfig || Yes || GRPO configuration
|-
| train_dataset || Dataset || Yes || Dataset with "prompt" column
|-
| processing_class || PreTrainedTokenizer || Yes || Tokenizer for generation
|-
| reward_funcs || List[Callable] || Yes || Reward functions to score completions
|-
| eval_dataset || Dataset || No || Evaluation dataset
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| TrainOutput || namedtuple || Training metrics including rewards, KL divergence
|-
| model || (side effect) || Model weights updated via policy gradient
|}

== Usage Examples ==

=== Basic GRPO Training ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Load model with vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,  # REQUIRED for GRPO
    gpu_memory_utilization=0.5,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    use_gradient_checkpointing="unsloth",
)

# Prepare dataset
dataset = Dataset.from_list([
    {"prompt": "What is 15% of 80?", "answer": "12"},
    {"prompt": "Solve: 2x + 5 = 15", "answer": "5"},
])

# Define reward function
def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer):
        if ans in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# Configure GRPO
grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,
    max_completion_length=512,
    beta=0.001,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    bf16=True,
    logging_steps=1,
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=[correctness_reward],
)

# Train
trainer.train()
</syntaxhighlight>

=== With SFT Warmup ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(model, r=64)

# Phase 1: SFT Warmup
sft_config = SFTConfig(
    output_dir="./sft_warmup",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=sft_dataset,
)
sft_trainer.train()

# Phase 2: GRPO Training
grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,
    max_completion_length=512,
    beta=0.001,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=3,
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=grpo_dataset,
    reward_funcs=[correctness_reward, format_reward],
)
grpo_trainer.train()
</syntaxhighlight>

=== Multiple Reward Functions ===
<syntaxhighlight lang="python">
# Define multiple reward aspects
def accuracy_reward(completions, answer, **kwargs):
    """Check answer correctness."""
    rewards = []
    for c, a in zip(completions, answer):
        rewards.append(1.0 if a in c else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Check for chain-of-thought format."""
    rewards = []
    for c in completions:
        if "<think>" in c and "</think>" in c:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def brevity_reward(completions, **kwargs):
    """Slight bonus for concise answers."""
    rewards = []
    for c in completions:
        # Reward shorter completions (within reason)
        length = len(c)
        if length < 200:
            rewards.append(0.1)
        elif length < 500:
            rewards.append(0.0)
        else:
            rewards.append(-0.1)
    return rewards

# Use all rewards
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=[
        accuracy_reward,
        format_reward,
        brevity_reward,
    ],
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_GRPO_Training]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]
* [[requires_env::Environment:Unslothai_Unsloth_VLLM]]
