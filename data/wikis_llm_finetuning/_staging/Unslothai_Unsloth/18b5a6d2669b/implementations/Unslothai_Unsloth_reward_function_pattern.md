# Implementation: reward_function_pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl/grpo_trainer]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Reward_Modeling]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Pattern specification for user-defined reward functions that score model completions during GRPO reinforcement learning training.

=== Description ===

This is a **Pattern Doc** - it documents a user-defined interface that must be implemented for GRPO training. Reward functions are the core mechanism for providing learning signal in reinforcement learning. They take model completions and return numerical scores indicating quality.

Key requirements:
* Input: Lists of completions and prompts
* Output: List of float scores (one per completion)
* Should be deterministic for training stability
* Can use heuristics, external models, or verification

=== Usage ===

Define one or more reward functions and pass them to GRPOTrainer via the `reward_funcs` parameter. Multiple reward functions can be combined (their scores are summed or weighted).

== Interface Specification ==

=== Required Signature ===
<syntaxhighlight lang="python">
def reward_function(
    completions: List[str],  # Model-generated completions
    prompts: List[str],      # Original prompts (for context)
    **kwargs                 # Optional: metadata, model outputs
) -> List[float]:            # Reward scores, one per completion
    """
    Score model completions for GRPO training.

    Args:
        completions: List of generated text strings
        prompts: List of corresponding input prompts
        **kwargs: Optional additional context

    Returns:
        List of float rewards (same length as completions)
        Higher scores = better completions
    """
    ...
</syntaxhighlight>

=== Score Guidelines ===
{| class="wikitable"
|-
! Property !! Guideline
|-
| Range || Typically -1 to 1, or 0 to 1
|-
| Scale || Consistent across examples
|-
| Meaning || Higher = better completion
|-
| Determinism || Prefer deterministic for stability
|}

== Usage Examples ==

=== Format Checking Reward (Rule-Based) ===
<syntaxhighlight lang="python">
import re

def format_reward(completions, prompts, **kwargs):
    """
    Reward for correct output format (reasoning + boxed answer).
    """
    rewards = []
    for completion in completions:
        score = 0.0

        # Check for reasoning tags
        if "<think>" in completion and "</think>" in completion:
            score += 0.5

        # Check for boxed answer
        if re.search(r"\\boxed\{.+\}", completion):
            score += 0.5

        rewards.append(score)

    return rewards
</syntaxhighlight>

=== Answer Correctness Reward ===
<syntaxhighlight lang="python">
import re

def correctness_reward(completions, prompts, **kwargs):
    """
    Reward for correct math answers.
    Requires ground truth answers in kwargs or dataset.
    """
    rewards = []
    answers = kwargs.get("answers", [None] * len(completions))

    for completion, answer in zip(completions, answers):
        if answer is None:
            rewards.append(0.0)
            continue

        # Extract model's answer from \boxed{}
        match = re.search(r"\\boxed\{(.+?)\}", completion)
        if match:
            model_answer = match.group(1).strip()
            # Compare (simple string match)
            score = 1.0 if model_answer == str(answer).strip() else 0.0
        else:
            score = 0.0

        rewards.append(score)

    return rewards
</syntaxhighlight>

=== Length Penalty Reward ===
<syntaxhighlight lang="python">
def length_reward(completions, prompts, max_length=500, **kwargs):
    """
    Penalize overly long or short completions.
    """
    rewards = []
    for completion in completions:
        length = len(completion)

        if length < 50:  # Too short
            score = -0.5
        elif length > max_length:  # Too long
            score = -0.3
        else:
            score = 0.0  # Good length range

        rewards.append(score)

    return rewards
</syntaxhighlight>

=== Model-Based Reward ===
<syntaxhighlight lang="python">
def model_reward(completions, prompts, reward_model, tokenizer, **kwargs):
    """
    Use a trained reward model to score completions.
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # Prepare input for reward model
        full_text = prompt + completion
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True)

        # Get reward score
        with torch.no_grad():
            score = reward_model(**inputs).logits.item()

        rewards.append(score)

    return rewards
</syntaxhighlight>

=== Combining Multiple Rewards ===
<syntaxhighlight lang="python">
from unsloth import UnslothGRPOTrainer, UnslothGRPOConfig

# Define multiple reward functions
def format_reward(completions, prompts, **kwargs):
    # ... format checking
    return format_scores

def correctness_reward(completions, prompts, **kwargs):
    # ... answer verification
    return correctness_scores

# Pass as list to trainer
trainer = UnslothGRPOTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    reward_funcs = [format_reward, correctness_reward],  # Combined
    args = UnslothGRPOConfig(...),
)

# Rewards are summed: total = format + correctness
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Reward_Functions]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment]]
