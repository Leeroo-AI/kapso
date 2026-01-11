# Principle: Reward_Functions

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|RLHF|https://arxiv.org/abs/2203.02155]]
* [[source::Paper|Constitutional AI|https://arxiv.org/abs/2212.08073]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Reward_Modeling]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Design and implementation of reward signals that guide reinforcement learning policy optimization toward desired model behaviors.

=== Description ===

Reward Functions are the mechanism by which we communicate objectives to RL algorithms. In GRPO (Group Relative Policy Optimization), rewards score multiple completions per prompt, enabling the algorithm to identify which behaviors to reinforce.

Types of reward functions:
1. **Rule-based**: Heuristic checks (format, length, keywords)
2. **Verification-based**: Programmatic correctness (math, code execution)
3. **Model-based**: Trained reward models or LLM-as-judge
4. **Hybrid**: Combination of multiple signals

Effective reward functions should be:
* Aligned with the true objective
* Informative (provide gradient signal)
* Robust to gaming/hacking

=== Usage ===

Design reward functions when:
* Training with GRPO, PPO, or other RL methods
* You can define success criteria for completions
* Ground truth or verification is available

Key design decisions:
* What behaviors to reward/penalize
* How to weight multiple objectives
* Whether to use sparse or dense rewards

== Theoretical Basis ==

=== GRPO Objective ===

GRPO optimizes:

<math>
\mathcal{L}_{GRPO} = -\mathbb{E}_{x \sim D} \mathbb{E}_{y_1, \ldots, y_G \sim \pi_\theta(\cdot|x)} \left[ \sum_{i=1}^{G} (r_i - \bar{r}) \log \pi_\theta(y_i|x) \right]
</math>

Where:
* G completions are generated per prompt x
* r_i is the reward for completion y_i
* r̄ is the mean reward (group reference)
* The advantage (r_i - r̄) determines if behavior is reinforced or suppressed

=== Reward Shaping ===

Dense rewards provide more learning signal than sparse rewards:

'''Sparse:''' Score = 1 if correct, 0 otherwise
'''Dense:''' Score = 0.5 for format + 0.5 × similarity(answer, target)

Dense rewards guide learning more effectively but risk reward hacking.

=== Reward Hacking ===

Models may find unintended ways to maximize reward:

<math>
\pi^* = \arg\max_\pi \mathbb{E}_{\pi}[r] \neq \pi_{desired}
</math>

Mitigations:
* Multiple orthogonal reward functions
* KL divergence penalty from reference model
* Human evaluation checkpoints

=== Multi-Objective Rewards ===

Combining rewards:

<math>
r_{total} = \sum_{i} w_i \cdot r_i
</math>

Or using reward stacking (GRPOTrainer sums rewards automatically):

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Reward function design (abstract)
def design_reward(task_type):
    if task_type == "math":
        rewards = [
            format_reward,        # Check for reasoning structure
            correctness_reward,   # Verify answer
        ]
    elif task_type == "code":
        rewards = [
            syntax_reward,        # Valid code
            execution_reward,     # Runs correctly
            style_reward,         # Code quality
        ]
    elif task_type == "general":
        rewards = [
            model_reward,         # LLM-as-judge
            length_reward,        # Appropriate length
        ]

    return rewards
</syntaxhighlight>

=== Reward Scaling ===

Rewards should be normalized for stable training:
* Mean-centered: subtract running average
* Scaled: divide by standard deviation
* Clipped: limit extreme values

GRPO's group-relative formulation provides automatic normalization within each batch.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_reward_function_pattern]]

