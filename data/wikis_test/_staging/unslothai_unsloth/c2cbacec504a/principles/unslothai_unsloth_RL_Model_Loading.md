# Principle: unslothai_unsloth_RL_Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO: Group Relative Policy Optimization|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|DeepSeekMath|https://arxiv.org/abs/2402.03300]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for loading language models with fast inference capabilities specifically optimized for reinforcement learning training loops.

=== Description ===

Reinforcement Learning model loading differs from standard SFT loading because:

1. **Fast generation is critical**: RL training loops generate many completions per step
2. **Higher capacity adapters**: RL often benefits from higher LoRA rank (64+)
3. **Memory partitioning**: GPU memory must be split between model weights and inference cache
4. **Continuous batching**: Efficient handling of variable-length generation

The key insight is that RL training is bottlenecked by generation speed, not training speed, making vLLM integration essential.

=== Usage ===

Use this principle when:
- Setting up GRPO, PPO, or other RL algorithms
- Training requires sampling multiple completions per prompt
- Generation speed is the primary bottleneck

== Theoretical Basis ==

=== RL Training Loop Structure ===

The RL training loop has fundamentally different characteristics:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# SFT Training Loop (generation-free)
for batch in dataloader:
    loss = model.forward(batch)  # One forward pass
    loss.backward()
    optimizer.step()

# RL Training Loop (generation-heavy)
for batch in dataloader:
    # GENERATION PHASE (bottleneck!)
    prompts = batch["prompt"]
    completions = []
    for _ in range(num_generations):  # e.g., 8 completions per prompt
        completion = model.generate(prompts, max_new_tokens=256)
        completions.append(completion)

    # REWARD PHASE
    rewards = reward_function(completions, prompts)

    # TRAINING PHASE
    loss = compute_grpo_loss(model, completions, rewards)
    loss.backward()
    optimizer.step()
</syntaxhighlight>

Without vLLM, the generation phase dominates training time (70-90%).

=== vLLM Integration Benefits ===

vLLM provides:

1. **PagedAttention**: Efficient KV cache management
2. **Continuous Batching**: Process multiple sequences simultaneously
3. **Optimized Kernels**: Flash attention, fused operations

<syntaxhighlight lang="python">
# Speed comparison (approximate)
# Standard HF generate: 50 tokens/second
# vLLM generate: 200-500 tokens/second

# For GRPO with 8 generations × 256 tokens × 32 batch:
# HF: ~1,310 seconds per step
# vLLM: ~130-330 seconds per step
# Speedup: 4-10x
</syntaxhighlight>

=== Memory Partitioning ===

GPU memory must be carefully allocated:

<syntaxhighlight lang="python">
# 24GB GPU example
total_vram = 24  # GB

# Model weights (4-bit 7B model)
model_weights = 3.5  # GB

# Training tensors (gradients, optimizer)
training_overhead = 8  # GB

# vLLM KV cache
# gpu_memory_utilization controls this
vllm_allocation = total_vram * 0.5  # 12 GB at 0.5 util

# Available for training
remaining = total_vram - vllm_allocation  # 12 GB
# Must fit: model_weights + training_overhead
</syntaxhighlight>

=== Higher LoRA Rank for RL ===

RL benefits from higher capacity adapters:

<syntaxhighlight lang="python">
# SFT: Lower rank usually sufficient
# Learning direct mappings from demonstrations
sft_rank = 16  # Good for most SFT tasks

# RL: Higher rank helps learn from rewards
# Model must explore and refine based on signal
rl_rank = 64  # Recommended for GRPO

# Why? RL updates are noisier than SFT
# Higher rank provides more degrees of freedom
# to capture subtle reward-correlated behaviors
</syntaxhighlight>

=== Fast Inference Mode Architecture ===

<syntaxhighlight lang="python">
# When fast_inference=True:
# 1. Model is loaded normally with LoRA
# 2. vLLM engine is attached for generation
# 3. Training uses PyTorch, generation uses vLLM

model_architecture = {
    "base_model": "4-bit quantized weights",
    "lora_adapters": "16-bit trainable",
    "vllm_engine": {
        "kv_cache": "paged attention",
        "scheduler": "continuous batching",
        "sampling": "parallel token generation"
    }
}
</syntaxhighlight>

== Practical Guide ==

=== Recommended Settings by GPU ===

{| class="wikitable"
|-
! GPU !! Model Size !! gpu_memory_utilization !! max_lora_rank
|-
| RTX 4090 (24GB) || 3B || 0.5-0.6 || 64
|-
| A100 (40GB) || 7B || 0.5-0.6 || 64-128
|-
| A100 (80GB) || 13B+ || 0.6-0.7 || 128
|}

=== Troubleshooting vLLM Issues ===

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM during generation | vLLM cache too large | Lower `gpu_memory_utilization` |
| Slow generation | Cache too small | Increase `gpu_memory_utilization` |
| vLLM not found | Missing dependency | `pip install vllm` |
| LoRA not applied | Rank mismatch | Check `max_lora_rank` ≥ actual rank |

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastLanguageModel_from_pretrained_vllm]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
