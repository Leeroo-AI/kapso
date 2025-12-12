{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|DeepSeekMath: GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|TRL GRPOTrainer|https://huggingface.co/docs/trl]]
* [[source::Doc|Unsloth GRPO Guide|https://docs.unsloth.ai/]]
|-
! Domains
| [[domain::RLHF]], [[domain::Reasoning]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Reinforcement learning technique that improves reasoning by training on groups of sampled responses with relative rewards.

=== Description ===
Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm designed to improve model reasoning capabilities. Unlike DPO which uses static preference pairs, GRPO generates multiple responses per prompt online and computes relative rewards within each group. This approach, popularized by DeepSeek for mathematical reasoning, provides richer training signals and adapts to the model's current capabilities.

=== Usage ===
Use this principle when you want to improve model reasoning abilities (math, coding, logic). Apply when you can define a reward function or have a verifier for response quality. Particularly effective for chain-of-thought improvement and self-consistency enhancement. Requires more compute than DPO but can achieve stronger reasoning performance.

== Theoretical Basis ==
'''Algorithm Overview:'''
1. For each prompt x, sample G responses: \{y_1, ..., y_G\}
2. Compute rewards: \{r_1, ..., r_G\}
3. Normalize rewards within group: \(\hat{r}_i = (r_i - \mu_G) / \sigma_G\)
4. Update policy using normalized rewards

'''GRPO Objective:'''
\[
\mathcal{L}_{GRPO} = -\mathbb{E}_{x \sim D, y_i \sim \pi_\theta} \left[ \hat{r}_i \cdot \log \pi_\theta(y_i|x) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]
\]

'''Group Normalization:'''
Key innovation - rewards are normalized within each group:
\[
\hat{r}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\}) + \epsilon}
\]

This provides:
* Baseline estimation without separate value network
* Adaptive difficulty - harder prompts have higher variance
* Reduced reward hacking

'''Pseudo-code:'''
<syntaxhighlight lang="python">
def grpo_step(model, ref_model, prompts, reward_fn, G=4, beta=0.1):
    """
    One GRPO training step
    """
    all_losses = []
    
    for prompt in prompts:
        # Generate G responses
        responses = model.generate(prompt, num_return_sequences=G)
        
        # Compute rewards
        rewards = [reward_fn(prompt, r) for r in responses]
        
        # Normalize within group
        rewards = (rewards - mean(rewards)) / (std(rewards) + 1e-8)
        
        # Compute policy gradient loss
        for response, reward in zip(responses, rewards):
            log_prob = model.log_prob(response | prompt)
            ref_log_prob = ref_model.log_prob(response | prompt)
            
            kl_penalty = beta * (log_prob - ref_log_prob)
            loss = -reward * log_prob + kl_penalty
            all_losses.append(loss)
    
    return mean(all_losses)
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:TRL_GRPOTrainer]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]
* [[uses_heuristic::Heuristic:Batch_Size_Optimization]]

