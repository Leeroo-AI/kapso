{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|RL Guide|https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Reward_Engineering]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of defining reward functions that guide model behavior during reinforcement learning training.

=== Description ===

Reward functions determine what behaviors are reinforced:

**Reward Types:**
1. **Rule-based**: Check format, length, keywords
2. **Model-based**: Use reward model to score
3. **Outcome-based**: Verify correctness (math, code)
4. **Composite**: Combine multiple signals

**Design Principles:**
- Clear signal (rewards should differentiate good vs bad)
- Avoid reward hacking
- Balance multiple objectives

== Practical Guide ==

=== Basic Reward Function ===
<syntaxhighlight lang="python">
def reward_function(completions, prompts):
    """
    Returns list of rewards (one per completion).
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        reward = 0.0

        # Format rewards
        if "<think>" in completion and "</think>" in completion:
            reward += 1.0

        # Answer format
        if "\\boxed{" in completion or "Answer:" in completion:
            reward += 1.0

        # Length penalty
        if len(completion.split()) < 10:
            reward -= 1.0

        rewards.append(reward)
    return rewards
</syntaxhighlight>

=== Math Correctness Reward ===
<syntaxhighlight lang="python">
import re

def math_reward(completions, prompts, ground_truths):
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        # Extract answer from \boxed{...}
        match = re.search(r'\\boxed\{([^}]+)\}', completion)
        if match:
            answer = match.group(1).strip()
            if answer == str(gt).strip():
                rewards.append(1.0)  # Correct
            else:
                rewards.append(-0.5)  # Wrong answer
        else:
            rewards.append(-1.0)  # No answer
    return rewards
</syntaxhighlight>

=== Reward Model Integration ===
<syntaxhighlight lang="python">
from transformers import pipeline

reward_model = pipeline("text-classification", model="reward-model")

def model_reward(completions, prompts):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        full_text = prompt + completion
        score = reward_model(full_text)[0]["score"]
        rewards.append(score)
    return rewards
</syntaxhighlight>

=== Composite Reward ===
<syntaxhighlight lang="python">
def composite_reward(completions, prompts):
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # Format: 0-2 points
        format_score = check_format(completion)

        # Correctness: 0-5 points
        correct_score = check_correctness(completion, prompt)

        # Length: -1 to 0 points
        length_penalty = length_score(completion)

        # Weighted combination
        total = 0.3 * format_score + 0.6 * correct_score + 0.1 * length_penalty
        rewards.append(total)
    return rewards
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_PatchFastRL]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
