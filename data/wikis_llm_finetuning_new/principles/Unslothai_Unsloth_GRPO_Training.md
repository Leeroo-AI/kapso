# Principle: GRPO_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|DeepSeekMath: Pushing the Limits of Mathematical Reasoning|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl/grpo_trainer]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::NLP]], [[domain::Training]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for training language models using Group Relative Policy Optimization, which improves reasoning through reward-guided generation.

=== Description ===

GRPO (Group Relative Policy Optimization) is an on-policy reinforcement learning algorithm designed for improving reasoning capabilities in language models. Unlike PPO, GRPO avoids the need for a separate value model by using group-relative reward normalization.

Key characteristics:
* **Group Normalization**: Rewards are normalized within groups of completions for the same prompt
* **No Value Model**: Simplifies training compared to PPO
* **On-Policy**: Generates completions from current policy during training
* **KL Regularization**: Prevents policy from diverging too far from reference model

GRPO is particularly effective for math reasoning, code generation, and tasks with verifiable outputs.

=== Usage ===

Use this principle when:
* Training reasoning models for math, coding, or logic tasks
* Outputs can be evaluated by programmatic reward functions
* You want to improve chain-of-thought reasoning
* After SFT warmup to further optimize model behavior

This is the core RL training step, following model setup and reward function design.

== Theoretical Basis ==

'''GRPO Objective:'''
<math>
\mathcal{L}_{GRPO} = -\mathbb{E}_{x, y \sim \pi_\theta} \left[ \hat{A}(x, y) \cdot \log \pi_\theta(y|x) \right] + \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})
</math>

Where:
- π_θ is the policy being trained
- π_ref is the reference (initial) policy
- Â(x, y) is the group-normalized advantage
- β is the KL penalty coefficient

'''Group Relative Advantage:'''
For each prompt x with G generated completions {y_1, ..., y_G}:

<math>
\hat{A}(x, y_i) = \frac{r(x, y_i) - \text{mean}_{j}(r(x, y_j))}{\text{std}_{j}(r(x, y_j)) + \epsilon}
</math>

This normalizes rewards within each group, making training more stable.

'''Training Loop:'''
<syntaxhighlight lang="python">
# Pseudo-code for GRPO training
for batch in dataset:
    # Generate G completions per prompt using vLLM
    completions = vllm_generate(batch.prompts, num_generations=G)

    # Compute rewards
    rewards = reward_function(completions)

    # Normalize within groups
    advantages = group_normalize(rewards)

    # Compute policy gradient loss
    logprobs = model.forward(completions)
    policy_loss = -mean(advantages * logprobs)

    # Add KL penalty
    kl_div = compute_kl(model, ref_model, completions)
    total_loss = policy_loss + beta * kl_div

    # Update
    total_loss.backward()
    optimizer.step()
</syntaxhighlight>

'''Key Hyperparameters:'''
{| class="wikitable"
|-
! Parameter !! Typical Value !! Effect
|-
| num_generations || 4-8 || More = better advantage estimates, more compute
|-
| beta || 0.001-0.1 || Lower = more policy change, higher = more conservative
|-
| temperature || 0.7-1.0 || Higher = more diverse generations
|-
| max_completion_length || 256-1024 || Depends on task complexity
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_GRPOTrainer_train]]

