# Principle: Reward_Function_Design

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|Training Language Models to Follow Instructions|https://arxiv.org/abs/2203.02155]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::NLP]], [[domain::Reward_Modeling]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for defining reward functions that score model completions to guide reinforcement learning training.

=== Description ===

Reward functions are the core feedback mechanism in RL training. They take model-generated completions and return scalar scores indicating quality. GRPO uses these rewards to compute policy gradients that improve generation quality.

Reward functions can be:
1. **Rule-Based**: Programmatic checks (format compliance, correctness verification)
2. **Model-Based**: Neural network reward models trained on human preferences
3. **Hybrid**: Combination of rules and models

For math/reasoning tasks, rule-based rewards checking answer correctness are highly effective. For open-ended generation, model-based rewards better capture quality.

=== Usage ===

Use this principle when:
* Defining what "good" output looks like for RL training
* The task has verifiable correctness criteria
* You need to balance multiple quality aspects
* Training reasoning models with chain-of-thought

This step comes after dataset preparation and before GRPO training.

== Theoretical Basis ==

'''Reward Function Interface:'''
<math>
R: \mathcal{Y} \times \mathcal{X} \rightarrow \mathbb{R}
</math>

Where:
- Y is the space of completions
- X is the space of prompts
- R(y, x) is the scalar reward for completion y given prompt x

'''GRPO Reward Normalization:'''
GRPO normalizes rewards within groups for stable training:

<math>
\hat{r}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}
</math>

Where r is the vector of rewards for completions from the same prompt.

'''Multiple Reward Combination:'''
<syntaxhighlight lang="python">
# Combine multiple reward signals
total_reward = (
    correctness_reward * 1.0 +      # Is the answer correct?
    format_reward * 0.5 +            # Does it follow format?
    length_penalty * 0.1             # Penalize excessive length
)
</syntaxhighlight>

'''Key Design Principles:'''
1. **Sparse vs Dense**: Sparse rewards (correct/incorrect) vs dense rewards (partial credit)
2. **Reward Shaping**: Add intermediate rewards to guide learning
3. **Reward Hacking**: Models may exploit reward functions in unintended ways
4. **Normalization**: Rewards should be on consistent scale

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_Reward_Function_Interface]]

