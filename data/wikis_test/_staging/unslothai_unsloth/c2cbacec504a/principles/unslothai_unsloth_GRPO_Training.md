# Principle: unslothai_unsloth_GRPO_Training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|DeepSeekMath: Pushing Math Reasoning|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|PPO: Proximal Policy Optimization|https://arxiv.org/abs/1707.06347]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Policy_Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for training language models using group-relative policy optimization where multiple completions per prompt are compared against each other.

=== Description ===

GRPO (Group Relative Policy Optimization) is an RL algorithm that:

1. **Samples multiple completions** per prompt (typically 8)
2. **Computes rewards** for each completion
3. **Calculates group-relative advantages** (completion vs group mean)
4. **Updates policy** to increase probability of higher-reward completions

Key advantage over PPO: GRPO doesn't require a separate value/critic network, reducing memory and complexity.

=== Usage ===

Use GRPO when:
- Training models on tasks with verifiable rewards (math, code)
- Optimizing for complex objectives beyond imitation
- You have access to reward signals but not demonstrations
- You want to improve reasoning without human preference data

== Theoretical Basis ==

=== GRPO Objective ===

The GRPO loss function:

<math>
\mathcal{L}_{GRPO} = -\mathbb{E}_{q \sim D} \left[ \frac{1}{G} \sum_{i=1}^{G} A_i \cdot \log \pi_\theta(o_i | q) - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref}) \right]
</math>

Where:
- <math>G</math>: Group size (number of completions per prompt)
- <math>A_i</math>: Advantage of completion i
- <math>\pi_\theta</math>: Current policy
- <math>\pi_{ref}</math>: Reference policy (initial model)
- <math>\beta</math>: KL penalty coefficient

=== Advantage Computation ===

Group-relative advantage:

<math>
A_i = \frac{r_i - \bar{r}}{\sigma_r}
</math>

<syntaxhighlight lang="python">
def compute_advantages(rewards):
    """
    Compute group-relative advantages.
    rewards: [batch_size, num_generations]
    """
    # Mean reward per prompt
    mean_reward = rewards.mean(dim=1, keepdim=True)

    # Advantage: how much better than group average
    advantages = rewards - mean_reward

    # Normalize for stable training
    std = rewards.std(dim=1, keepdim=True)
    advantages = advantages / (std + 1e-8)

    return advantages
</syntaxhighlight>

=== KL Divergence Regularization ===

Prevents policy from deviating too far from reference:

<syntaxhighlight lang="python">
def compute_kl_penalty(log_probs_current, log_probs_ref):
    """
    KL divergence between current and reference policy.
    """
    # Per-token KL
    kl_per_token = log_probs_current - log_probs_ref

    # Sum over sequence
    kl_sequence = kl_per_token.sum(dim=-1)

    return kl_sequence
</syntaxhighlight>

=== GRPO vs PPO ===

{| class="wikitable"
|-
! Aspect !! PPO !! GRPO
|-
| Value network || Required || Not required
|-
| Memory usage || Higher || Lower
|-
| Baseline || Learned value || Group mean
|-
| Variance || Lower (learned) || Higher (empirical)
|-
| Best for || General RL || LLM fine-tuning
|}

=== Training Loop ===

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
def grpo_training_step(model, batch, reward_func, ref_model):
    prompts = batch["prompt"]

    # 1. Generate multiple completions
    completions = []
    for _ in range(num_generations):
        comp = model.generate(prompts, max_new_tokens=256)
        completions.append(comp)
    # completions: [num_gen, batch_size, seq_len]

    # 2. Compute rewards
    rewards = []
    for comp in completions:
        r = reward_func(comp, prompts)
        rewards.append(r)
    # rewards: [num_gen, batch_size]

    # 3. Compute advantages
    advantages = compute_advantages(rewards)

    # 4. Compute policy gradient loss
    log_probs = model.forward(completions).log_probs
    ref_log_probs = ref_model.forward(completions).log_probs

    policy_loss = -(advantages * log_probs).mean()
    kl_loss = kl_coef * compute_kl_penalty(log_probs, ref_log_probs)

    total_loss = policy_loss + kl_loss

    # 5. Update
    total_loss.backward()
    optimizer.step()

    return total_loss
</syntaxhighlight>

== Practical Guide ==

=== Hyperparameter Guidelines ===

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `num_generations` | 4-16 | More = better advantage estimate, slower |
| `learning_rate` | 1e-6 to 5e-6 | Much lower than SFT |
| `kl_coef` | 0.01 to 0.1 | Higher = more conservative |
| `max_completion_length` | 128-512 | Task-dependent |

=== Common Issues ===

| Issue | Symptom | Solution |
|-------|---------|----------|
| Reward hacking | Model exploits reward bugs | Fix reward function |
| KL divergence explosion | Model collapses | Increase `kl_coef` |
| No improvement | Flat rewards | Check reward variance |
| Mode collapse | Same outputs | Lower learning rate |

=== Monitoring Training ===

<syntaxhighlight lang="python">
# Key metrics to track
metrics = {
    "reward/mean": "Average reward (should increase)",
    "reward/std": "Reward variance (should stay reasonable)",
    "kl_divergence": "Should stay bounded",
    "policy_loss": "Main training signal",
}
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_GRPOTrainer_train]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
