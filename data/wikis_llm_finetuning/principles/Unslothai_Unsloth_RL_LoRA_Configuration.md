# Principle: RL_LoRA_Configuration

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|PPO|https://arxiv.org/abs/1707.06347]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Reinforcement_Learning]], [[domain::Parameter_Efficient_Finetuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Configuration of LoRA adapters for reinforcement learning training, with specific constraints for vLLM integration and capacity requirements for policy optimization.

=== Description ===

LoRA Configuration for RL differs from standard SFT in several key aspects:

1. **Higher ranks typical** - RL often requires r=32-64 vs r=8-16 for SFT, as policy optimization needs more model capacity to learn from reward signals
2. **vLLM rank constraint** - LoRA rank must not exceed `max_lora_rank` specified during model loading, as vLLM pre-allocates memory
3. **Same target modules** - Typically all attention + MLP projections

The policy network in RL must be flexible enough to explore the reward landscape while being constrained by the LoRA rank bottleneck.

=== Usage ===

Apply RL LoRA Configuration when:
* Setting up GRPO, PPO, or DPO training
* Model was loaded with `fast_inference=True`
* Need higher capacity than standard SFT

Key constraint: `r <= max_lora_rank` from model loading

== Theoretical Basis ==

=== Capacity Requirements ===

RL training optimizes:

<math>
J(\theta) = \mathbb{E}_{\pi_\theta}[R(s, a)]
</math>

Where π_θ is the policy parameterized by LoRA weights θ. Higher rank provides more capacity for:
* Exploring diverse policy behaviors
* Fitting to complex reward landscapes
* Capturing nuanced generation patterns

=== Rank and vLLM Memory ===

vLLM pre-allocates for maximum rank:

<math>
\text{LoRA Memory} = 2 \times r_{max} \times d \times n_{layers} \times \text{sizeof(dtype)}
</math>

Using actual rank r < r_max doesn't save memory in vLLM, but:
* Reduces compute per forward pass
* May improve generalization (regularization)

=== Policy Gradient with LoRA ===

Gradient flows only through LoRA parameters:

<math>
\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]
</math>

Where A_t is the advantage. LoRA's low-rank structure provides implicit regularization against overfitting to reward hacking.

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# RL LoRA configuration (abstract)
def configure_lora_for_rl(model, max_lora_rank):
    # Determine rank based on task complexity
    # Typical: r=32-64 for GRPO, r=16-32 for DPO

    if task_complexity == "high":
        r = max_lora_rank  # Use full capacity
    else:
        r = max_lora_rank // 2  # Balance capacity vs regularization

    # Apply LoRA to all key modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    return apply_lora(model, r=r, targets=target_modules)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_get_peft_model_rl]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection_Tip]]
