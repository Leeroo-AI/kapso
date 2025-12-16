{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|DeepSeekMath: Pushing the Limits of Mathematical Reasoning|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|Direct Preference Optimization|https://arxiv.org/abs/2305.18290]]
* [[source::Paper|Proximal Policy Optimization Algorithms|https://arxiv.org/abs/1707.06347]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Deep_Learning]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Reinforcement learning algorithm that optimizes language model policies using group-relative advantage estimation, eliminating the need for a separate critic/value model while enabling efficient training of reasoning capabilities.

=== Description ===
Group Relative Policy Optimization (GRPO) is a simplified alternative to PPO (Proximal Policy Optimization) designed specifically for language model fine-tuning. Introduced by DeepSeek for training mathematical reasoning models, GRPO achieves comparable or better results with significantly less memory and compute.

Key innovations of GRPO:
1. **No Critic Model**: Instead of learning a value function, GRPO estimates advantages from the group of sampled responses
2. **Group-Relative Baseline**: For each prompt, generate N responses and use their mean reward as baseline
3. **Memory Efficiency**: ~80% less VRAM than PPO since no value model is maintained
4. **Online Learning**: Generates fresh completions during training rather than using a fixed dataset

The algorithm is particularly effective for:
- Mathematical reasoning (GSM8K, AIME, MATH)
- Code generation and debugging
- Structured output generation
- Any task with verifiable reward signals

GRPO naturally handles sparse rewards (correct/incorrect) and can combine multiple reward functions for multi-objective optimization.

=== Usage ===
Use GRPO training when:
- Teaching reasoning capabilities to language models
- You have clear reward signals (format compliance, answer correctness)
- Standard SFT underperforms due to need for exploration
- PPO is too memory-intensive for your hardware

Prerequisites for effective GRPO:
- A model already capable of basic instruction following (SFT warm-up recommended)
- Well-defined reward functions that capture desired behavior
- Sufficient compute for generating multiple responses per prompt
- Tasks where correctness can be programmatically verified

== Theoretical Basis ==
GRPO optimizes a clipped surrogate objective similar to PPO, but estimates advantages differently.

'''Policy Gradient Objective:'''
<math>
\mathcal{L}_{GRPO} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \hat{A}, \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}, 1-\epsilon, 1+\epsilon\right) \hat{A}\right)\right]
</math>

Where the advantage <math>\hat{A}</math> is computed using group-relative estimation.

'''Group-Relative Advantage Estimation:'''
<syntaxhighlight lang="python">
# Pseudo-code for GRPO advantage computation
def compute_group_advantages(prompt, model, reward_funcs, num_generations=8):
    """
    Generate multiple responses and compute advantages relative to group.
    """
    # Generate N responses for this prompt
    responses = []
    for _ in range(num_generations):
        response = model.generate(prompt)
        responses.append(response)

    # Compute rewards for each response
    rewards = []
    for response in responses:
        r = sum(reward_fn(prompt, response) for reward_fn in reward_funcs)
        rewards.append(r)

    # Normalize: advantage = (reward - mean) / std
    mean_reward = mean(rewards)
    std_reward = std(rewards) + epsilon

    advantages = [(r - mean_reward) / std_reward for r in rewards]

    return responses, advantages
</syntaxhighlight>

'''Comparison to PPO:'''
<syntaxhighlight lang="text">
PPO:
┌─────────────────┐     ┌─────────────────┐
│  Policy Model   │     │  Value Model    │
│  (generates)    │     │  (estimates V)  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         v                       v
    Response r              Value V(s)
         │                       │
         └───────────┬───────────┘
                     v
              Advantage = R - V(s)

GRPO:
┌─────────────────┐
│  Policy Model   │
│  (generates N)  │
└────────┬────────┘
         │
         v
   Responses [r1, r2, ..., rN]
         │
         v
   Rewards [R1, R2, ..., RN]
         │
         v
   Advantage = (Ri - mean(R)) / std(R)
</syntaxhighlight>

'''Training Loop:'''
<syntaxhighlight lang="python">
# Pseudo-code for GRPO training
def grpo_training_step(model, prompt_batch, reward_funcs, config):
    """
    One step of GRPO training.
    """
    all_responses = []
    all_advantages = []
    all_old_logprobs = []

    # Generate and compute advantages for each prompt
    for prompt in prompt_batch:
        responses, advantages = compute_group_advantages(
            prompt, model, reward_funcs, config.num_generations
        )
        old_logprobs = compute_logprobs(model, prompt, responses)

        all_responses.extend(responses)
        all_advantages.extend(advantages)
        all_old_logprobs.extend(old_logprobs)

    # Policy gradient update
    for epoch in range(config.ppo_epochs):
        new_logprobs = compute_logprobs(model, prompts, all_responses)
        ratio = exp(new_logprobs - all_old_logprobs)

        # Clipped surrogate objective
        surr1 = ratio * all_advantages
        surr2 = clip(ratio, 1-config.clip, 1+config.clip) * all_advantages
        loss = -mean(min(surr1, surr2))

        loss.backward()
        optimizer.step()
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel]]
* [[implemented_by::Implementation:unslothai_unsloth_get_peft_model]]

=== Tips and Tricks ===
