# Implementation: FastLanguageModel_from_pretrained_vllm

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for loading language models with vLLM fast inference enabled for reinforcement learning provided by Unsloth.

=== Description ===

This is `FastLanguageModel.from_pretrained` configured with `fast_inference=True` to enable vLLM backend. When loaded this way, the model includes a `vllm_engine` attribute for high-throughput generation during RL training.

Key vLLM parameters:
* `gpu_memory_utilization`: Fraction of GPU memory for vLLM KV cache
* `max_lora_rank`: Maximum LoRA rank that vLLM can handle for adapter hot-swapping
* `float8_kv_cache`: Enable FP8 quantization for KV cache (reduces memory)
* `disable_log_stats`: Suppress vLLM logging output

=== Usage ===

Import and use when setting up GRPO, PPO, or other RL training workflows. The `fast_inference=True` flag is required for TRL's RL trainers to use vLLM generation. Without it, training will fall back to slow HuggingFace generation.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py
* '''Lines:''' 121-676 (same as standard, but with fast_inference path)

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def from_pretrained(
    model_name: str = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    # ... standard parameters ...

    # vLLM-specific parameters (relevant when fast_inference=True)
    fast_inference: bool = False,  # Set to True for RL
    gpu_memory_utilization: float = 0.5,  # vLLM GPU memory fraction
    float8_kv_cache: bool = False,  # FP8 KV cache quantization
    random_state: int = 3407,
    max_lora_rank: int = 64,  # Max LoRA rank for vLLM
    disable_log_stats: bool = True,  # Suppress vLLM logging
    *args,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model with optional vLLM backend for RL training.

    When fast_inference=True:
    - Creates vLLM engine attached as model.vllm_engine
    - Enables fast generation during RL training
    - Supports LoRA adapter hot-swapping via vLLM

    Args:
        fast_inference: Enable vLLM backend (REQUIRED for GRPO/RL)
        gpu_memory_utilization: Fraction of GPU for vLLM (0.0-1.0)
        float8_kv_cache: Use FP8 for KV cache (saves memory)
        max_lora_rank: Maximum LoRA rank vLLM will support
        disable_log_stats: Suppress vLLM statistics logging

    Returns:
        Tuple of (model, tokenizer) with model.vllm_engine available
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model_name || str || Yes || HuggingFace model ID or local path
|-
| max_seq_length || int || No || Maximum sequence length (default: 2048)
|-
| load_in_4bit || bool || No || Enable 4-bit quantization (default: True)
|-
| fast_inference || bool || '''Yes (for RL)''' || Enable vLLM backend (must be True for GRPO)
|-
| gpu_memory_utilization || float || No || vLLM GPU memory fraction (default: 0.5)
|-
| float8_kv_cache || bool || No || Enable FP8 KV cache (default: False)
|-
| max_lora_rank || int || No || Maximum LoRA rank for vLLM (default: 64)
|-
| disable_log_stats || bool || No || Suppress vLLM logging (default: True)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Model with vllm_engine attribute for fast generation
|-
| tokenizer || PreTrainedTokenizer || Configured tokenizer
|-
| model.vllm_engine || vLLM Engine || vLLM engine for high-throughput generation
|}

== Usage Examples ==

=== Basic GRPO Setup ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with vLLM for GRPO training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    fast_inference=True,  # REQUIRED for GRPO
    gpu_memory_utilization=0.5,
    max_lora_rank=64,
)

# Verify vLLM engine is available
print(f"vLLM engine: {hasattr(model, 'vllm_engine')}")
</syntaxhighlight>

=== Memory-Optimized Loading ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with FP8 KV cache for reduced memory
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,  # More memory for larger batches
    float8_kv_cache=True,  # FP8 KV cache saves memory
    max_lora_rank=64,
    disable_log_stats=True,
)
</syntaxhighlight>

=== High-Rank LoRA for RL ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Higher max_lora_rank for more complex adaptation
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.5,
    max_lora_rank=128,  # Higher rank for RL adaptation
)

# Then use higher rank in get_peft_model
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Must be <= max_lora_rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_RL_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_11]]
* [[requires_env::Environment:Unslothai_Unsloth_VLLM]]
