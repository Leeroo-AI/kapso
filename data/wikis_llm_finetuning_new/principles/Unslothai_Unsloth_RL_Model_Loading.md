# Principle: RL_Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for loading language models with vLLM fast inference backend enabled for reinforcement learning training workflows.

=== Description ===

RL Model Loading extends standard QLoRA model loading with vLLM integration for high-throughput generation during RL training. Unlike supervised fine-tuning where generation is only needed at inference time, RL training requires generating completions during training to compute rewards and policy gradients.

The vLLM backend provides:
* PagedAttention for efficient KV cache management
* Continuous batching for high throughput
* Optimized CUDA kernels for generation
* FP8 KV cache support for memory efficiency

When `fast_inference=True`, the model maintains both:
1. A trainable model for gradient computation
2. A vLLM engine (`model.vllm_engine`) for fast generation

=== Usage ===

Use this principle when:
* Training with GRPO, PPO, DPO, or other RL algorithms
* Requiring fast generation during the training loop
* The training process involves on-policy generation
* Memory permits running both training model and vLLM engine

This is the first step in GRPO/RL workflows, requiring `fast_inference=True`.

== Theoretical Basis ==

'''vLLM PagedAttention:'''
Instead of allocating contiguous memory for KV cache, PagedAttention stores keys and values in non-contiguous blocks:

<math>
\text{Memory}_{vLLM} = \frac{\text{Memory}_{standard}}{k} + \text{overhead}
</math>

Where k depends on the waste reduction from dynamic allocation.

'''Generation During Training:'''
<syntaxhighlight lang="python">
# Pseudo-code for RL training loop
for batch in dataset:
    # Generate completions using vLLM (fast)
    with model.for_inference():
        completions = model.vllm_engine.generate(batch.prompts)

    # Compute rewards
    rewards = reward_function(completions)

    # Compute policy gradient loss
    model.for_training()
    loss = compute_policy_loss(batch, completions, rewards)
    loss.backward()
</syntaxhighlight>

'''Key Differences from Standard Loading:'''
{| class="wikitable"
|-
! Aspect !! Standard QLoRA !! RL Loading
|-
| fast_inference || False || True (required)
|-
| vllm_engine || Not created || Attached to model
|-
| Memory usage || Lower || Higher (dual model)
|-
| Generation speed || Slow (HF generate) || Fast (vLLM)
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm]]

