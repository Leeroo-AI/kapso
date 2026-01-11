# Principle: RL_Model_Loading

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
* [[source::Paper|vLLM|https://arxiv.org/abs/2309.06180]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Reinforcement_Learning]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Technique for loading language models with high-throughput inference backends that enable efficient generation sampling required for reinforcement learning training.

=== Description ===

RL Model Loading prepares a language model for reinforcement learning workflows like GRPO (Group Relative Policy Optimization) or PPO (Proximal Policy Optimization). Unlike standard SFT, RL training requires generating multiple completions per prompt during training:

* **GRPO**: 6-16 completions per prompt for group-relative reward comparison
* **PPO**: Multiple trajectories for advantage estimation

This makes inference throughput critical to training speed. vLLM integration enables:
* Continuous batching for variable-length generation
* PagedAttention for efficient KV cache management
* LoRA serving without full model duplication

=== Usage ===

Apply RL Model Loading when:
* Training with GRPO, PPO, or other on-policy RL algorithms
* Generation throughput is a training bottleneck
* Multiple completions per prompt are needed

Prerequisites:
* vLLM installed and functional
* Sufficient GPU memory for model + vLLM KV cache
* CUDA-capable GPU

== Theoretical Basis ==

=== Generation Bottleneck in RL ===

For on-policy RL, each training step requires:

<math>
\text{Time}_{step} = \text{Time}_{generate} + \text{Time}_{reward} + \text{Time}_{update}
</math>

With standard HuggingFace generation, `Time_generate` dominates due to:
* Sequential token generation
* No batching across different prompts
* Memory-inefficient KV cache

=== vLLM Optimizations ===

vLLM reduces generation time through:

1. **Continuous Batching**: Process tokens from different requests together
2. **PagedAttention**: Non-contiguous KV cache with virtual memory
3. **Speculative Decoding**: Draft model for faster token acceptance

Memory equation for vLLM:

<math>
\text{GPU}_{total} = \text{GPU}_{model} + \text{GPU}_{utilization} \times \text{GPU}_{available}
</math>

Where `gpu_memory_utilization` controls KV cache allocation.

=== LoRA + vLLM Integration ===

vLLM supports serving LoRA adapters with:
* Pre-allocation based on `max_lora_rank`
* Efficient adapter weight loading
* No model recompilation for rank changes

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# RL model loading (abstract)
def load_for_rl(model_name, max_lora_rank):
    # Load base model with quantization
    model = load_quantized(model_name)

    # Initialize vLLM engine
    vllm_engine = vLLM(
        model = model,
        max_lora_rank = max_lora_rank,  # Pre-allocate for LoRA
        gpu_memory_utilization = 0.5,   # Reserve for KV cache
        enable_lora = True,
    )

    # Attach engine to model
    model.vllm_engine = vllm_engine
    model.generate = vllm_engine.generate  # Override generate

    return model
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_FastLanguageModel_from_pretrained_vllm]]

