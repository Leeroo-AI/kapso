{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|Training Language Models to Follow Instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|Constitutional AI: Harmlessness from AI Feedback|https://arxiv.org/abs/2212.08073]]
* [[source::Paper|Scaling Laws for Reward Model Overoptimization|https://arxiv.org/abs/2210.10760]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::NLP]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Design pattern for creating scalar reward signals that guide reinforcement learning optimization of language models, combining format compliance, correctness verification, and quality metrics into a unified training objective.

=== Description ===
Reward functions are the core mechanism for specifying desired behavior in RL-based language model training. Unlike supervised learning where target outputs are provided, RL uses reward signals to indicate the quality of generated responses. Well-designed reward functions are critical for successful GRPO/PPO training.

Reward function design involves:
1. **Decomposition**: Break the desired behavior into measurable components
2. **Scaling**: Assign appropriate magnitudes to each reward component
3. **Verification**: Ensure rewards accurately reflect desired outcomes
4. **Combination**: Aggregate multiple signals without conflicting gradients

Common reward components for LLM training:
- **Format compliance**: Does the output follow expected structure (XML tags, JSON, etc.)?
- **Correctness**: Is the answer factually/mathematically correct?
- **Reasoning quality**: Does the reasoning chain lead logically to the answer?
- **Length penalty**: Discourage overly verbose or terse responses
- **Safety/Alignment**: Penalize harmful or off-topic content

The reward signal directly shapes what the model learns - poorly designed rewards lead to reward hacking where models exploit loopholes rather than learning intended behavior.

=== Usage ===
Design reward functions when:
- Training models with GRPO, PPO, or other RL algorithms
- Tasks have verifiable success criteria (math, code, structured output)
- Multiple quality dimensions need simultaneous optimization
- Standard SFT doesn't capture nuanced quality differences

Reward design guidelines:
- Start with sparse, binary rewards (correct/incorrect)
- Add shaping rewards gradually to guide exploration
- Test for reward hacking on held-out examples
- Normalize rewards to similar scales across components

== Theoretical Basis ==
Reward functions map (prompt, response) pairs to scalar values that the RL algorithm maximizes:

<math>
R: (p, r) \rightarrow \mathbb{R}
</math>

'''Multi-Component Reward Design:'''
<syntaxhighlight lang="python">
# Pseudo-code for composite reward function
def composite_reward(prompt, response, expected_answer):
    """
    Combine multiple reward signals with weighted sum.
    """
    rewards = {}

    # Format compliance (0 or positive)
    rewards["format"] = check_format(response) * 3.0

    # Answer correctness (negative to positive)
    correctness = check_answer(response, expected_answer)
    if correctness == "exact_match":
        rewards["correctness"] = 3.0
    elif correctness == "close":
        rewards["correctness"] = 1.0
    else:
        rewards["correctness"] = -1.5

    # Length penalty (small negative for extremes)
    length = len(response)
    if length < 10:
        rewards["length"] = -0.5
    elif length > 2000:
        rewards["length"] = -0.5
    else:
        rewards["length"] = 0.0

    # Total reward
    return sum(rewards.values())
</syntaxhighlight>

'''Format Verification:'''
<syntaxhighlight lang="python">
# Pseudo-code for format checking
def check_format(response, pattern=None):
    """
    Verify response follows expected structure.

    Example pattern for reasoning tasks:
    <reasoning>...</reasoning>
    <answer>...</answer>
    """
    import re

    if pattern is None:
        # Default structured reasoning format
        pattern = r"^[\s]*<reasoning>.+?</reasoning>.*?<answer>.+?</answer>[\s]*$"

    if re.match(pattern, response, re.DOTALL):
        return 1.0  # Full credit for correct format
    elif "<reasoning>" in response and "<answer>" in response:
        return 0.5  # Partial credit for elements present
    else:
        return 0.0  # No credit for wrong format
</syntaxhighlight>

'''Answer Extraction and Verification:'''
<syntaxhighlight lang="python">
# Pseudo-code for answer checking
def check_answer(response, expected):
    """
    Extract and verify answer from structured response.
    """
    import re

    # Extract answer from tags
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not match:
        return "missing"

    extracted = match.group(1).strip()
    expected = str(expected).strip()

    # Clean numeric answers
    extracted_clean = re.sub(r"[%$,]", "", extracted)
    expected_clean = re.sub(r"[%$,]", "", expected)

    # Exact match
    if extracted_clean == expected_clean:
        return "exact_match"

    # Numeric tolerance for floating point
    try:
        ratio = float(extracted_clean) / float(expected_clean)
        if 0.95 <= ratio <= 1.05:
            return "close"
    except (ValueError, ZeroDivisionError):
        pass

    return "wrong"
</syntaxhighlight>

'''Reward Scaling Principles:'''
<syntaxhighlight lang="text">
Reward Magnitude Guidelines:
┌─────────────────────────────────────────────────────┐
│ Reward Type        │ Typical Range │ Rationale     │
├─────────────────────────────────────────────────────┤
│ Format compliance  │ 0 to +3       │ Necessary but │
│                    │               │ not sufficient│
├─────────────────────────────────────────────────────┤
│ Correct answer     │ +2 to +5      │ Primary goal  │
├─────────────────────────────────────────────────────┤
│ Wrong answer       │ -1 to -2      │ Discourage    │
│                    │               │ but don't     │
│                    │               │ over-penalize │
├─────────────────────────────────────────────────────┤
│ Length penalty     │ -0.5 to 0     │ Minor shaping │
└─────────────────────────────────────────────────────┘

Key: Keep rewards balanced so no single component dominates.
Large negative rewards can destabilize training.
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]

=== Tips and Tricks ===
