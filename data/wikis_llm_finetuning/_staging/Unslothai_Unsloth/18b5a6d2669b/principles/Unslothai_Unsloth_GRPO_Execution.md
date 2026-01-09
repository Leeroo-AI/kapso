# Principle: GRPO_Execution

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|PPO|https://arxiv.org/abs/1707.06347]]
* [[source::Paper|DeepSeekMath|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Policy_Optimization]], [[domain::NLP]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Execution of Group Relative Policy Optimization training loop, iteratively generating completions, computing rewards, and updating the policy network.

=== Description ===

GRPO Execution implements the training loop for Group Relative Policy Optimization:

1. **Sample prompts** from the training dataset
2. **Generate completions** using vLLM (G per prompt)
3. **Compute rewards** via user-defined reward functions
4. **Calculate group-relative advantages** (r_i - mean(r))
5. **Update policy** with weighted log-probability gradients
6. **Repeat** until convergence

GRPO simplifies PPO by eliminating the value network and using within-group baselines.

=== Usage ===

Execute GRPO training after configuring model, reward functions, and dataset. Monitor:
* Mean reward progression
* KL divergence from reference
* Generation quality (sample outputs)

== Theoretical Basis ==

=== GRPO Training Loop ===

<math>
\text{For each batch:}
</math>

1. Sample prompts: x_1, ..., x_B
2. Generate: y_{i,1}, ..., y_{i,G} ~ π_θ(·|x_i) for each prompt
3. Reward: r_{i,j} = R(y_{i,j}, x_i)
4. Advantage: A_{i,j} = r_{i,j} - (1/G)Σ_k r_{i,k}
5. Loss: L = -Σ_{i,j} A_{i,j} · log π_θ(y_{i,j}|x_i) + β·KL
6. Update: θ ← θ - η·∇L

=== Group-Relative Baseline ===

The key innovation of GRPO is computing the baseline within each group:

<math>
\bar{r}_i = \frac{1}{G}\sum_{j=1}^G r_{i,j}
</math>

This provides:
* Automatic normalization (mean-centered rewards)
* No value network training
* Lower variance than global baselines

=== KL Divergence Regularization ===

To prevent catastrophic forgetting:

<math>
D_{KL}(\pi_\theta || \pi_{ref}) = \mathbb{E}_{\pi_\theta}\left[\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right]
</math>

The reference model (π_ref) is typically the SFT checkpoint.

=== Training Dynamics ===

GRPO exhibits:
* **Initial exploration**: Rewards vary widely
* **Policy sharpening**: Model converges on rewarded behaviors
* **KL constraint binding**: Policy stabilizes near reference

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# GRPO training loop (abstract)
def grpo_train(model, dataset, reward_funcs, num_generations, beta):
    for batch in dataset:
        prompts = batch["prompt"]

        # Generate G completions per prompt
        with model.inference_mode():
            completions = model.generate(
                prompts,
                num_return_sequences=num_generations
            )

        # Compute rewards
        rewards = sum(rf(completions, prompts) for rf in reward_funcs)
        rewards = reshape(rewards, (batch_size, num_generations))

        # Group-relative advantages
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - baseline

        # Compute loss
        log_probs = model.log_prob(completions, prompts)
        policy_loss = -(advantages * log_probs).mean()
        kl_loss = beta * compute_kl(model, ref_model, completions)
        loss = policy_loss + kl_loss

        # Update
        loss.backward()
        optimizer.step()
</syntaxhighlight>

=== Convergence Indicators ===

Training is progressing well when:
* Mean reward increases steadily
* Reward variance decreases
* KL divergence stabilizes (not exploding)
* Sample generations improve qualitatively

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_UnslothGRPOTrainer]]

