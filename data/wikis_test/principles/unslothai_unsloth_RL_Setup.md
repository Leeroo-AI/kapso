{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|PPO: Proximal Policy Optimization|https://arxiv.org/abs/1707.06347]]
* [[source::Paper|DPO: Direct Preference Optimization|https://arxiv.org/abs/2305.18290]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Deep_Learning]], [[domain::Policy_Optimization]], [[domain::Reasoning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Configuration and optimization technique for reinforcement learning fine-tuning of Large Language Models, enabling memory-efficient training with policy gradient methods.

=== Description ===

RL Setup in Unsloth prepares the training environment for policy optimization algorithms:

**Key Challenges Addressed:**
1. **Memory overhead**: RL requires storing model states for both generation and training
2. **Sampling efficiency**: Generating multiple completions per prompt is computationally expensive
3. **Gradient computation**: Policy gradients require different memory patterns than SFT

**Unsloth Solutions:**
- vLLM integration for fast batch generation (10-50x faster than HF generate)
- Optimized gradient computation reduces memory by 80%
- Automatic mode switching between inference and training
- Fused operations for reward computation

**Supported Algorithms:**
- GRPO: Group Relative Policy Optimization (recommended for reasoning)
- PPO: Proximal Policy Optimization
- DPO/ORPO/KTO: Preference-based methods

=== Usage ===

Apply RL setup when:
* Enhancing model reasoning capabilities (math, coding, logic)
* Aligning models with human preferences
* Training models to follow complex instructions
* Optimizing for specific output patterns (format, style)

Requirements:
- GPU with 24GB+ VRAM (for vLLM colocate mode)
- vLLM installed (`pip install vllm`)
- TRL library >= 0.12.0

== Theoretical Basis ==

=== Policy Gradient Methods ===

RL training optimizes a policy π to maximize expected reward:

<math>
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t R(s_t, a_t) \right]
</math>

The policy gradient is:

<math>
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t) \right]
</math>

Where A(s,a) is the advantage function.

=== GRPO: Group Relative Policy Optimization ===

GRPO simplifies advantage estimation by using relative rewards within a group:

<syntaxhighlight lang="python">
# GRPO advantage computation
def compute_grpo_advantage(completions, rewards, prompt):
    """
    For each prompt, generate G completions.
    Advantage = reward - mean(rewards_in_group)
    """
    group_rewards = rewards  # All rewards for this prompt

    # Normalize within group
    mean_reward = group_rewards.mean()
    std_reward = group_rewards.std() + 1e-8

    advantages = (group_rewards - mean_reward) / std_reward

    return advantages
</syntaxhighlight>

GRPO loss:
<math>
L_{GRPO} = -\mathbb{E} \left[ \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \cdot A(x,y) - \beta \cdot KL(\pi_\theta || \pi_{ref}) \right]
</math>

=== Memory-Efficient RL Training ===

Standard RL requires storing:
1. Reference model for KL computation
2. Current policy for generation
3. Current policy for gradient update

Unsloth optimization:

<syntaxhighlight lang="python">
# Abstract memory-efficient RL loop
def efficient_rl_step(model, prompts, reward_fn):
    # Phase 1: Generation (inference mode, no gradients)
    with inference_mode():
        # vLLM handles generation efficiently
        completions = vllm_generate(model, prompts, num_samples=8)

    # Phase 2: Reward computation (no model forward)
    rewards = reward_fn(completions, prompts)

    # Phase 3: Policy update (training mode)
    with training_mode():
        # Only forward pass for sampled completions
        # Not the full generation process
        log_probs = model.forward(completions).log_probs

        # Compute loss with KL penalty
        loss = compute_grpo_loss(log_probs, rewards, ref_log_probs)
        loss.backward()

    return loss
</syntaxhighlight>

=== vLLM Integration ===

vLLM provides efficient generation through:
- PagedAttention: KV cache paging for variable-length sequences
- Continuous batching: Dynamic batch formation
- Tensor parallelism: Multi-GPU generation

<syntaxhighlight lang="python">
# Abstract vLLM setup for RL
def setup_vllm_for_rl(model, gpu_memory_utilization=0.6):
    """
    Configure vLLM to coexist with training.

    gpu_memory_utilization < 1.0 reserves memory for gradients.
    """
    vllm_config = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_lora_rank": 64,  # Match LoRA rank
        "enable_lora": True,
    }

    # vLLM takes model weights, Unsloth manages LoRA
    return vLLMEngine(vllm_config)
</syntaxhighlight>

=== Inference/Training Mode Switching ===

<syntaxhighlight lang="python">
# Abstract mode switching
class ModelStateSwitcher:
    def for_inference(self, model):
        """Prepare model for fast generation."""
        model.eval()
        # Disable dropout
        # Enable inference optimizations
        # Switch to vLLM backend

    def for_training(self, model):
        """Prepare model for gradient updates."""
        model.train()
        # Enable dropout
        # Enable gradient computation
        # Use standard PyTorch forward
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_PatchFastRL]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_RL_Hyperparameters]]
