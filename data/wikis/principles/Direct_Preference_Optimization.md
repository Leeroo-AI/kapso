{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|DPO: Direct Preference Optimization|https://arxiv.org/abs/2305.18290]]
* [[source::Doc|TRL DPOTrainer|https://huggingface.co/docs/trl/dpo_trainer]]
* [[source::Blog|Understanding DPO|https://huggingface.co/blog/dpo-trl]]
|-
! Domains
| [[domain::RLHF]], [[domain::Alignment]], [[domain::LLMs]]
|-
! Last Updated
| [[last_updated::2025-12-12 00:00 GMT]]
|}

== Overview ==
Alignment technique that trains language models directly on preference pairs without explicit reward modeling, simplifying the RLHF pipeline.

=== Description ===
Direct Preference Optimization (DPO) provides a simpler alternative to PPO-based RLHF. Instead of training a separate reward model and using reinforcement learning, DPO directly optimizes the policy using preference pairs (chosen vs. rejected responses). It derives a closed-form loss from the RL objective, making alignment more stable and efficient.

=== Usage ===
Use this principle after SFT when you want to align model outputs with human preferences. Apply when you have preference data (pairs of good and bad responses to the same prompt). Preferred over PPO when training stability is important and you don't need online learning.

== Theoretical Basis ==
'''Key Insight:'''
The optimal policy under the RLHF objective has a closed form:

\[
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
\]

This can be rearranged to express reward in terms of policies:

\[
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
\]

'''DPO Loss:'''
Substituting into the Bradley-Terry preference model:

\[
\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
\]

Where:
* \(y_w\) = chosen/winning response
* \(y_l\) = rejected/losing response
* \(\beta\) = temperature controlling deviation from reference
* \(\pi_{ref}\) = reference policy (usually the SFT model)

'''Simplified:'''
<syntaxhighlight lang="python">
def dpo_loss(pi_logprobs_chosen, pi_logprobs_rejected,
             ref_logprobs_chosen, ref_logprobs_rejected, beta):
    """
    Compute DPO loss
    """
    pi_logratios = pi_logprobs_chosen - pi_logprobs_rejected
    ref_logratios = ref_logprobs_chosen - ref_logprobs_rejected
    
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    return loss
</syntaxhighlight>

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:TRL_DPOTrainer]]

=== Tips and Tricks ===
* [[uses_heuristic::Heuristic:Learning_Rate_Tuning]]

