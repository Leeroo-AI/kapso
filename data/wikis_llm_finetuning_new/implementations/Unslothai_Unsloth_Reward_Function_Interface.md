# Implementation: Reward_Function_Interface

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::NLP]], [[domain::Reward_Modeling]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Pattern documentation for user-defined reward functions in GRPO training.

=== Description ===

This is a **Pattern Doc** describing the interface that user-defined reward functions must implement for GRPOTrainer. Reward functions receive batches of completions and return corresponding reward scores.

The GRPOTrainer supports multiple reward functions that can be combined. Each function should focus on one aspect of quality (correctness, format, length, etc.).

=== Usage ===

Define one or more reward functions following this interface and pass them to GRPOTrainer's `reward_funcs` parameter. Functions can be as simple as regex matching or as complex as calling external models.

== Interface Specification ==

=== Function Signature ===
<syntaxhighlight lang="python">
def reward_function(
    completions: List[str],
    prompts: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Score a batch of completions.

    Args:
        completions: List of generated text completions
        prompts: List of original prompts (optional, for context)
        **kwargs: Additional context (e.g., answer key, metadata)

    Returns:
        List of float rewards, one per completion.
        Higher values indicate better completions.
    """
    rewards = []
    for completion in completions:
        score = compute_score(completion)  # Your logic here
        rewards.append(score)
    return rewards
</syntaxhighlight>

=== GRPOTrainer Usage ===
<syntaxhighlight lang="python">
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=[
        correctness_reward,  # Your reward function
        format_reward,       # Another reward function
    ],
)
</syntaxhighlight>

== Example Implementations ==

=== Math Correctness Reward ===
<syntaxhighlight lang="python">
import re

def math_correctness_reward(
    completions: list[str],
    answer: list[str],  # Ground truth answers
    **kwargs,
) -> list[float]:
    """
    Reward function for math problem correctness.

    Returns 1.0 for correct answers, 0.0 for incorrect.
    """
    rewards = []
    for completion, correct_answer in zip(completions, answer):
        # Extract final answer from completion
        # Look for patterns like "answer is X" or "= X" or just the number at end
        patterns = [
            r"(?:answer|result|solution)\s*(?:is|=|:)\s*([+-]?\d+\.?\d*)",
            r"=\s*([+-]?\d+\.?\d*)\s*$",
            r"([+-]?\d+\.?\d*)\s*$",
        ]

        extracted = None
        for pattern in patterns:
            match = re.search(pattern, completion, re.IGNORECASE)
            if match:
                extracted = match.group(1)
                break

        # Compare with ground truth
        try:
            if extracted is not None:
                is_correct = float(extracted) == float(correct_answer)
            else:
                is_correct = False
        except ValueError:
            is_correct = str(extracted) == str(correct_answer)

        rewards.append(1.0 if is_correct else 0.0)

    return rewards
</syntaxhighlight>

=== Format Compliance Reward ===
<syntaxhighlight lang="python">
def format_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """
    Reward function for chain-of-thought format compliance.

    Checks for <think>...</think> tags around reasoning.
    """
    rewards = []
    for completion in completions:
        # Check for thinking tags
        has_think_open = "<think>" in completion
        has_think_close = "</think>" in completion

        if has_think_open and has_think_close:
            # Proper format
            score = 1.0
        elif has_think_open or has_think_close:
            # Partial format (missing one tag)
            score = 0.5
        else:
            # No format
            score = 0.0

        rewards.append(score)

    return rewards
</syntaxhighlight>

=== Length Penalty Reward ===
<syntaxhighlight lang="python">
def length_penalty_reward(
    completions: list[str],
    max_length: int = 1000,
    **kwargs,
) -> list[float]:
    """
    Penalize excessively long completions.

    Returns 0.0 for completions under max_length,
    negative values for longer ones.
    """
    rewards = []
    for completion in completions:
        length = len(completion)
        if length <= max_length:
            score = 0.0  # No penalty
        else:
            # Gradual penalty for excess length
            excess = length - max_length
            score = -0.001 * excess  # -0.001 per excess character

        rewards.append(score)

    return rewards
</syntaxhighlight>

=== Combined Rewards in GRPOTrainer ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

# Load model with vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(model, r=64)

# Define reward functions
def correctness_reward(completions, answer, **kwargs):
    # ... (as above)
    pass

def format_reward(completions, **kwargs):
    # ... (as above)
    pass

# Configure GRPO
grpo_config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,
    max_completion_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
)

# Create trainer with multiple rewards
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=grpo_config,
    train_dataset=grpo_dataset,
    reward_funcs=[
        correctness_reward,  # Primary reward
        format_reward,       # Secondary reward
    ],
)

trainer.train()
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| completions || List[str] || Yes || Batch of generated completions to score
|-
| prompts || List[str] || No || Original prompts (for context)
|-
| **kwargs || Any || No || Additional context (answer, metadata, etc.)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| rewards || List[float] || Scalar reward for each completion (higher = better)
|}

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Reward_Function_Design]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_TRL]]
