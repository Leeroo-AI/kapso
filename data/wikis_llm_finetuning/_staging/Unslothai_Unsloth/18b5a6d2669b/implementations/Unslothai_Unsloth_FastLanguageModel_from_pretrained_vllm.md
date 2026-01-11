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
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for loading Large Language Models with vLLM fast inference backend enabled for reinforcement learning training workflows, provided by Unsloth.

=== Description ===

`FastLanguageModel.from_pretrained` with `fast_inference=True` loads a model with vLLM engine integration for high-throughput generation during RL training. vLLM enables efficient batch generation with continuous batching and PagedAttention, critical for GRPO training which requires multiple completions per prompt.

Key capabilities for RL training:
* **vLLM engine attachment** - Model gains `model.vllm_engine` for fast generation
* **Higher LoRA rank support** - `max_lora_rank` parameter for vLLM LoRA
* **GPU memory partitioning** - `gpu_memory_utilization` controls vLLM cache allocation
* **Continuous batching** - Efficient generation of multiple completions per prompt

This is an angle-specific documentation of `from_pretrained` for the RL Model Loading principle.

=== Usage ===

Use when setting up GRPO, PPO, or other RL training workflows that require fast generation sampling. The vLLM backend enables generating 6-16 completions per prompt efficiently, which is essential for GRPO's group-relative optimization.

NOT for:
* Standard SFT training (use default `fast_inference=False`)
* Systems without vLLM installed
* Very limited GPU memory (vLLM requires additional VRAM for KV cache)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py
* '''Lines:''' L121-700 (same function, different parameters)

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def from_pretrained(
    model_name: str = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    # RL-specific parameters
    fast_inference: bool = True,  # Enable vLLM
    gpu_memory_utilization: float = 0.5,  # vLLM GPU fraction
    max_lora_rank: int = 64,  # Maximum LoRA rank for vLLM
    float8_kv_cache: bool = False,  # FP8 KV cache for vLLM
    disable_log_stats: bool = True,
    # Standard parameters
    token: Optional[str] = None,
    use_gradient_checkpointing: str = "unsloth",
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model with vLLM fast inference backend for RL training.

    Args:
        model_name: HuggingFace model ID or local path
        max_seq_length: Maximum sequence length
        load_in_4bit: Enable 4-bit quantization
        fast_inference: Enable vLLM backend (True for RL)
        gpu_memory_utilization: Fraction of GPU for vLLM (0.0-1.0)
        max_lora_rank: Maximum LoRA rank supported by vLLM
        float8_kv_cache: Use FP8 for KV cache (memory optimization)

    Returns:
        Tuple of (model, tokenizer) with model.vllm_engine attached
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
| model_name || str || Yes || HuggingFace model ID
|-
| fast_inference || bool || Yes (True) || Must be True to enable vLLM
|-
| gpu_memory_utilization || float || No (default: 0.5) || Fraction of GPU for vLLM KV cache (0.0-1.0)
|-
| max_lora_rank || int || No (default: 64) || Maximum LoRA rank; must be >= actual LoRA rank
|-
| max_seq_length || int || No (default: 2048) || Maximum sequence length for generation
|-
| load_in_4bit || bool || No (default: True) || Enable 4-bit quantization
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PreTrainedModel || Model with `model.vllm_engine` for fast generation
|-
| tokenizer || PreTrainedTokenizer || Tokenizer with padding configured
|}

== Usage Examples ==

=== vLLM-Enabled Loading for GRPO ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model with vLLM for GRPO training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,          # Enable vLLM
    max_lora_rank = 64,             # Support up to rank 64 LoRA
    gpu_memory_utilization = 0.6,   # 60% GPU for vLLM cache
)

# Verify vLLM engine is attached
print(f"vLLM engine: {hasattr(model, 'vllm_engine')}")
</syntaxhighlight>

=== Higher Memory Utilization for Large Models ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load larger model with more GPU allocation to vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 32,              # Lower rank to fit in memory
    gpu_memory_utilization = 0.75,   # 75% GPU for vLLM
    float8_kv_cache = True,          # FP8 cache to save memory
)
</syntaxhighlight>

=== Complete GRPO Setup ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# 1. Load with vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.5,
)

# 2. Apply LoRA (rank must be <= max_lora_rank)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,  # <= max_lora_rank from loading
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

# 3. Configure chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# Model is now ready for GRPOTrainer
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_RL_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection_Tip]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Gradient_Checkpointing_Tip]]
