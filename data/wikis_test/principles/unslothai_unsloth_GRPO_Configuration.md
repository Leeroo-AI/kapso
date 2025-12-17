# Principle: unslothai_unsloth_GRPO_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL GRPO|https://huggingface.co/docs/trl/grpo_trainer]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Hyperparameter_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for selecting hyperparameters that control GRPO reinforcement learning training dynamics.

=== Description ===

GRPO Configuration involves balancing:
1. **Exploration vs Exploitation**: num_generations, temperature
2. **Stability vs Speed**: learning_rate, kl_coef
3. **Memory vs Quality**: batch_size, max_length

Proper configuration is critical for stable RL training.

=== Usage ===

Use when setting up any GRPO training run to define training behavior.

== Theoretical Basis ==

=== Key Hyperparameters ===

'''num_generations''': More completions = better advantage estimates but slower training

<math>
\text{Variance}(A) \propto \frac{1}{G}
</math>

'''kl_coef''': Controls policy constraint

<math>
\mathcal{L} = \mathcal{L}_{policy} + \beta \cdot D_{KL}(\pi || \pi_{ref})
</math>

=== Recommended Settings ===

| Task Type | num_gen | learning_rate | kl_coef |
|-----------|---------|---------------|---------|
| Math/Reasoning | 8-16 | 5e-6 | 0.05 |
| Code | 4-8 | 1e-5 | 0.1 |
| General | 4-8 | 5e-6 | 0.1 |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_GRPOConfig]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
