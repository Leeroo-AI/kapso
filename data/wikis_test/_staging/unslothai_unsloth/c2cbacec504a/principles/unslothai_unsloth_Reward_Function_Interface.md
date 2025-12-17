# Principle: unslothai_unsloth_Reward_Function_Interface

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|InstructGPT: Training LMs to Follow Instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|Constitutional AI|https://arxiv.org/abs/2212.08073]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Reward_Modeling]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Design pattern for defining reward signals that guide reinforcement learning optimization of language model behavior.

=== Description ===

The Reward Function Interface defines how user-specified quality metrics communicate with RL training algorithms. Unlike supervised learning where labels are explicit, RL relies on reward signals to shape model behavior through:

1. **Scalar feedback**: Single number indicating quality of a completion
2. **Comparative learning**: Model learns which completions are better than others
3. **Policy optimization**: Gradients are computed to increase expected reward

The reward function is the "objective" that GRPO optimizes—it must capture what "good" means for your task.

=== Usage ===

Use this principle when:
- Designing reward signals for GRPO/PPO training
- Combining multiple quality metrics
- Creating verifiable rewards for math/code tasks
- Building preference-based reward models

== Theoretical Basis ==

=== Reward in GRPO ===

GRPO (Group Relative Policy Optimization) uses rewards to compute advantages:

<math>
A_i = r_i - \frac{1}{G} \sum_{j=1}^{G} r_j
</math>

Where:
- <math>A_i</math>: Advantage for completion i
- <math>r_i</math>: Reward for completion i
- <math>G</math>: Number of completions per prompt

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
def compute_grpo_advantage(rewards_per_prompt):
    """
    GRPO uses group-relative advantages.
    Completions are compared within their group.
    """
    # rewards_per_prompt: [G,] rewards for one prompt
    mean_reward = rewards_per_prompt.mean()

    # Advantage: how much better than average
    advantages = rewards_per_prompt - mean_reward

    # Normalize for stable training
    if advantages.std() > 0:
        advantages = advantages / advantages.std()

    return advantages
</syntaxhighlight>

=== Reward Signal Properties ===

Effective reward functions have these properties:

{| class="wikitable"
|-
! Property !! Description !! Why It Matters
|-
| **Informative** || Non-zero gradient signal || Sparse rewards are hard to learn from
|-
| **Bounded** || Limited range (e.g., [-1, 1]) || Prevents gradient explosion
|-
| **Aligned** || Captures true objective || Reward hacking otherwise
|-
| **Decomposable** || Can combine multiple signals || Multi-objective optimization
|}

=== Reward Types ===

'''Rule-Based (Verifiable):'''
<syntaxhighlight lang="python">
# Deterministic, no model required
def rule_based_reward(completion, answer):
    if extract_answer(completion) == answer:
        return 1.0
    return 0.0
</syntaxhighlight>

'''Model-Based (Learned):'''
<syntaxhighlight lang="python">
# Neural reward model
def model_based_reward(completion, prompt):
    inputs = tokenize(prompt + completion)
    score = reward_model(inputs)
    return score.item()
</syntaxhighlight>

'''Hybrid (Combined):'''
<syntaxhighlight lang="python">
def hybrid_reward(completion, prompt, answer):
    rule_score = rule_based_reward(completion, answer)
    model_score = model_based_reward(completion, prompt)
    return 0.7 * rule_score + 0.3 * model_score
</syntaxhighlight>

=== Reward Shaping ===

Raw rewards can be transformed for better learning:

<syntaxhighlight lang="python">
def shape_reward(raw_reward):
    # 1. Clipping (prevent outliers)
    clipped = max(-1.0, min(1.0, raw_reward))

    # 2. Scaling (adjust magnitude)
    scaled = clipped * reward_scale

    # 3. Baseline subtraction (reduce variance)
    adjusted = scaled - baseline

    return adjusted

# GRPO does baseline subtraction automatically
# via group-relative advantages
</syntaxhighlight>

=== Multi-Objective Rewards ===

Combining multiple objectives:

<math>
r_{combined} = \sum_{k=1}^{K} w_k \cdot r_k
</math>

<syntaxhighlight lang="python">
# Weighted sum approach
reward_functions = [
    (correctness_reward, 0.6),  # Primary objective
    (format_reward, 0.2),       # Secondary
    (length_reward, 0.1),       # Tertiary
    (safety_reward, 0.1),       # Constraint
]

def combined_reward(completion, **kwargs):
    total = 0.0
    for func, weight in reward_functions:
        total += weight * func(completion, **kwargs)
    return total
</syntaxhighlight>

== Practical Guide ==

=== Common Reward Design Patterns ===

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Binary** | Verifiable tasks | Correct = 1, Wrong = 0 |
| **Graded** | Quality spectrum | Score 0-1 based on criteria |
| **Penalty** | Constraint satisfaction | Deduct for violations |
| **Bonus** | Encourage behaviors | Add for desired features |

=== Debugging Reward Functions ===

<syntaxhighlight lang="python">
# Always test reward functions before training
def debug_reward_function(reward_func, test_cases):
    print("Testing reward function...")
    for prompt, completion, expected in test_cases:
        actual = reward_func([completion], prompts=[prompt])[0]
        status = "✓" if abs(actual - expected) < 0.1 else "✗"
        print(f"{status} Expected {expected}, got {actual:.2f}")
        print(f"   Completion: {completion[:50]}...")

# Example test cases
test_cases = [
    ("What is 2+2?", "The answer is \\boxed{4}", 1.0),
    ("What is 2+2?", "I don't know", -0.5),
]
</syntaxhighlight>

=== Avoiding Reward Hacking ===

Models can exploit reward functions in unintended ways:

| Hack | Cause | Prevention |
|------|-------|------------|
| Length gaming | Length-correlated reward | Add length penalty |
| Keyword stuffing | Keyword-based scoring | Semantic evaluation |
| Format without content | Format-only reward | Combine with correctness |
| Repetition | Token-level rewards | N-gram penalties |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_reward_function_pattern]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
