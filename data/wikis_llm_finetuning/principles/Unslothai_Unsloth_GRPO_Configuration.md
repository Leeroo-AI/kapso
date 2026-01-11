# Principle: GRPO_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|DeepSeekMath|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Optimization]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Selection and tuning of hyperparameters for Group Relative Policy Optimization training, balancing exploration, stability, and computational efficiency.

=== Description ===

GRPO (Group Relative Policy Optimization) is a simplified variant of PPO designed for language model training. Configuration involves:

1. **Generation parameters**: How many completions to sample, at what length
2. **Optimization parameters**: Learning rate, batch size, KL penalty
3. **Efficiency parameters**: Memory management, gradient accumulation

GRPO eliminates the need for a separate value network by using group-relative advantages computed within each batch.

=== Usage ===

Configure GRPO when:
* Training reasoning models (math, code)
* Aligning models with verifiable reward signals
* Preference optimization with process-based rewards

Key trade-offs:
* More generations → better advantage estimation but slower
* Higher beta → more stable but less exploration
* Larger batch → smoother gradients but more memory

== Theoretical Basis ==

=== GRPO Objective ===

<math>
\mathcal{L}_{GRPO} = -\mathbb{E}_{x}\mathbb{E}_{y_1,...,y_G \sim \pi_\theta}\left[\sum_{i=1}^G (r_i - \bar{r}) \log \pi_\theta(y_i|x)\right] + \beta D_{KL}(\pi_\theta || \pi_{ref})
</math>

Where:
* G = num_generations completions per prompt
* r_i = reward for completion i
* r̄ = mean reward (baseline)
* β = KL penalty coefficient

=== Group-Relative Advantage ===

Unlike PPO, GRPO computes advantages within each prompt group:

<math>
A_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j
</math>

Benefits:
* No value network needed
* Automatic reward normalization
* Reduced variance compared to global baselines

=== Key Hyperparameters ===

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| num_generations | 6-16 | More = better estimation, slower |
| beta | 0.01-0.2 | Higher = more conservative |
| learning_rate | 1e-6 - 5e-5 | Higher = faster but unstable |
| max_completion_length | 100-500 | Task-dependent |

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# GRPO hyperparameter selection (abstract)
def configure_grpo(task_type, gpu_memory):
    if task_type == "math_reasoning":
        config = {
            "num_generations": 16,      # More samples for hard tasks
            "max_completion_length": 500,
            "beta": 0.05,               # Allow exploration
            "learning_rate": 1e-5,
        }
    elif task_type == "instruction_following":
        config = {
            "num_generations": 8,
            "max_completion_length": 200,
            "beta": 0.1,                # Moderate constraint
            "learning_rate": 5e-6,
        }

    # Adjust for memory
    if gpu_memory < 24:
        config["num_generations"] //= 2
        config["gradient_accumulation_steps"] *= 2

    return config
</syntaxhighlight>

=== Computational Cost ===

Per training step:
<math>
\text{Cost} = G \times \text{Cost}_{generate} + G \times \text{Cost}_{reward} + \text{Cost}_{update}
</math>

Generation dominates cost, making vLLM integration critical.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_UnslothGRPOConfig]]

