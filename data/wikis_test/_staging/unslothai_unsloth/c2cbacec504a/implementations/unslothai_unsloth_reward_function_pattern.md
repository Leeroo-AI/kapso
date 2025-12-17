# Implementation: unslothai_unsloth_reward_function_pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL GRPO|https://huggingface.co/docs/trl/grpo_trainer]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Reward_Modeling]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Pattern documentation for implementing user-defined reward functions in GRPO reinforcement learning training.

=== Description ===

Reward functions are user-defined callables that score model completions to guide reinforcement learning. Unlike most Implementations which document library APIs, this is a **Pattern Doc** that specifies the interface users must implement.

The reward function is the critical component that defines what "good" completions look like in GRPO training.

=== Usage ===

You must implement reward functions when:
- Training with GRPO or PPO
- Defining task-specific quality metrics
- Combining multiple reward signals

This is NOT a library API—it's a pattern you implement yourself.

== Interface Specification ==

=== Required Signature ===
<syntaxhighlight lang="python">
def reward_function(
    completions: List[str],    # Generated completions (batch)
    prompts: Optional[List[str]] = None,  # Original prompts
    **kwargs                   # Additional context (e.g., ground truth answers)
) -> List[float]:              # Reward scores (one per completion)
    """
    Score model completions for GRPO training.

    Args:
        completions: Batch of generated text completions
        prompts: Corresponding prompts (if needed)
        **kwargs: Extra context like 'answer' for math problems

    Returns:
        List of float rewards, same length as completions.
        Higher values = better completions.
    """
    ...
</syntaxhighlight>

=== Constraints ===

1. **Return length must match input**: `len(rewards) == len(completions)`
2. **Must return floats**: Integer or bool will cause issues
3. **Scores should be bounded**: Normalized to [-1, 1] or [0, 1] works best
4. **Must be deterministic**: Same input should give same output
5. **Higher = better**: GRPO maximizes reward

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| completions || List[str] || Yes || Model-generated completions to score
|-
| prompts || List[str] || No || Original prompts (for context-dependent scoring)
|-
| answer || List[str] || No || Ground truth answers (for verification tasks)
|-
| **kwargs || Any || No || Task-specific additional data
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| rewards || List[float] || Reward score for each completion (same length as completions)
|}

== Example Implementations ==

=== Format Checking Reward ===
<syntaxhighlight lang="python">
def format_reward(completions, **kwargs):
    """
    Reward completions that follow expected format.
    Used for training models to use specific reasoning patterns.
    """
    rewards = []
    for completion in completions:
        score = 0.0

        # Check for thinking tags
        if "<think>" in completion and "</think>" in completion:
            score += 0.25

        # Check for boxed answer
        if "\\boxed{" in completion:
            score += 0.25

        # Penalize extremely short responses
        if len(completion) < 50:
            score -= 0.5

        # Penalize extremely long responses
        if len(completion) > 2000:
            score -= 0.25

        rewards.append(score)
    return rewards
</syntaxhighlight>

=== Correctness Verification Reward ===
<syntaxhighlight lang="python">
import re

def correctness_reward(completions, answer, **kwargs):
    """
    Reward completions that contain the correct answer.
    For math/reasoning tasks with verifiable answers.
    """
    rewards = []
    for completion, target in zip(completions, answer):
        # Extract answer from boxed format
        match = re.search(r'\\boxed\{([^}]+)\}', completion)
        if match:
            extracted = match.group(1).strip()
            # Compare to target
            if extracted == str(target).strip():
                rewards.append(1.0)  # Correct
            else:
                rewards.append(0.0)  # Wrong answer
        else:
            rewards.append(-0.5)  # No answer found
    return rewards
</syntaxhighlight>

=== Length-Based Reward ===
<syntaxhighlight lang="python">
def length_reward(completions, **kwargs):
    """
    Reward based on response length.
    Encourages concise or verbose responses.
    """
    rewards = []
    target_length = 200  # Optimal length

    for completion in completions:
        length = len(completion)
        # Gaussian-like reward centered at target
        deviation = abs(length - target_length) / target_length
        score = max(0, 1.0 - deviation)
        rewards.append(score)
    return rewards
</syntaxhighlight>

=== Combined Multi-Objective Reward ===
<syntaxhighlight lang="python">
def combined_reward(completions, answer, **kwargs):
    """
    Combine multiple reward signals with weights.
    """
    # Get individual rewards
    format_scores = format_reward(completions)
    correct_scores = correctness_reward(completions, answer)
    length_scores = length_reward(completions)

    # Weighted combination
    rewards = []
    for f, c, l in zip(format_scores, correct_scores, length_scores):
        combined = (
            0.2 * f +    # 20% format
            0.7 * c +    # 70% correctness
            0.1 * l      # 10% length
        )
        rewards.append(combined)
    return rewards
</syntaxhighlight>

=== Using with GRPOTrainer ===
<syntaxhighlight lang="python">
from trl import GRPOTrainer, GRPOConfig

# Define your reward functions
def my_reward(completions, **kwargs):
    return [len(c) / 1000 for c in completions]

# Pass as list to GRPOTrainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [my_reward],  # List of reward functions
    args = grpo_config,
    train_dataset = dataset,
)
</syntaxhighlight>

== Best Practices ==

| Practice | Reason |
|----------|--------|
| Normalize to [-1, 1] or [0, 1] | Stable training dynamics |
| Make rewards informative | Sparse rewards are hard to learn from |
| Combine multiple signals | Single metrics often miss nuances |
| Test on sample data first | Debug before long training runs |
| Log reward distributions | Monitor for collapse or saturation |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_Reward_Function_Interface]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
