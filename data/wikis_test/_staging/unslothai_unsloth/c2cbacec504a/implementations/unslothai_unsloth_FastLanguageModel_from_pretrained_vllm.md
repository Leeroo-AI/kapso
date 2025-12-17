# Implementation: unslothai_unsloth_FastLanguageModel_from_pretrained_vllm

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|vLLM|https://docs.vllm.ai]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for loading language models with vLLM fast inference backend for reinforcement learning workflows.

=== Description ===

This is the same `FastLanguageModel.from_pretrained` API but with `fast_inference=True`, which enables:

1. **vLLM backend**: Uses vLLM's optimized inference engine for generation
2. **Higher LoRA rank support**: `max_lora_rank` parameter for RL which often needs r=64+
3. **GPU memory management**: `gpu_memory_utilization` controls vLLM's memory allocation
4. **Fast sampling**: Continuous batching and PagedAttention for efficient generation

This mode is specifically for GRPO/PPO training where fast generation is critical for sampling completions.

=== Usage ===

Use this when:
- Setting up GRPO or PPO reinforcement learning training
- You need fast batch generation during training
- The training loop requires sampling multiple completions per prompt

NOT for:
- Standard QLoRA SFT (use regular `from_pretrained` instead)
- Inference-only deployment (use vLLM directly)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/loader.py
* '''Lines:''' L120-620 (same function, different parameters)
* '''vLLM Integration:''' unsloth/models/_utils.py

=== Signature ===
<syntaxhighlight lang="python">
class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str = "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        # vLLM-specific parameters
        fast_inference: bool = True,  # ENABLE vLLM
        gpu_memory_utilization: float = 0.5,
        max_lora_rank: int = 64,
        disable_log_stats: bool = True,
        # Other parameters
        token: Optional[str] = None,
        use_gradient_checkpointing: str = "unsloth",
        **kwargs,
    ) -> Tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load model with vLLM fast inference for RL training.

        Args:
            model_name: HuggingFace model ID or path
            max_seq_length: Maximum sequence length
            fast_inference: Set True to enable vLLM backend
            gpu_memory_utilization: Fraction of GPU memory for vLLM (0.0-1.0)
            max_lora_rank: Maximum LoRA rank (64+ recommended for RL)
            disable_log_stats: Disable vLLM's verbose logging

        Returns:
            Tuple of (model, tokenizer) with vLLM backend enabled
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
| fast_inference || bool || Yes (True) || Must be True to enable vLLM
|-
| gpu_memory_utilization || float || No (default: 0.5) || vLLM GPU memory fraction (0.0-1.0)
|-
| max_lora_rank || int || No (default: 64) || Maximum LoRA rank for RL training
|-
| max_seq_length || int || No (default: 2048) || Maximum sequence length
|-
| load_in_4bit || bool || No (default: True) || Use 4-bit quantization
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModel || Model with vLLM backend for fast generation
|-
| tokenizer || PreTrainedTokenizer || Associated tokenizer
|}

== Usage Examples ==

=== Basic vLLM Loading for GRPO ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with vLLM backend enabled
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,           # Enable vLLM
    max_lora_rank = 64,              # Higher rank for RL
    gpu_memory_utilization = 0.6,    # 60% for vLLM
)

# Check vLLM is enabled
print(f"Fast inference enabled: {hasattr(model, 'vllm_engine')}")
</syntaxhighlight>

=== Memory Allocation Strategy ===
<syntaxhighlight lang="python">
# Memory is split between:
# - Model weights (~40% at default)
# - vLLM KV cache and workspace (~60% at default)

# For 24GB GPU:
# gpu_memory_utilization = 0.5 → ~12GB for vLLM, ~12GB for model
# gpu_memory_utilization = 0.7 → ~17GB for vLLM, ~7GB for model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    fast_inference = True,
    gpu_memory_utilization = 0.5,  # Conservative for stability
    max_lora_rank = 64,
)
</syntaxhighlight>

=== Complete GRPO Setup ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# 1. Load with vLLM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.6,
)

# 2. Add high-rank LoRA for RL
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,  # Higher rank for RL expressiveness
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 64,
    use_gradient_checkpointing = "unsloth",
)

# 3. Model is now ready for GRPOTrainer
print(model.print_trainable_parameters())
</syntaxhighlight>

=== vLLM Sampling Parameters ===
<syntaxhighlight lang="python">
from unsloth import vLLMSamplingParams

# Configure sampling for generation
sampling_params = vLLMSamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 256,
)

# These params are used during GRPO generation phase
</syntaxhighlight>

== vLLM vs Standard Mode ==

| Aspect | Standard Mode | vLLM Mode (`fast_inference=True`) |
|--------|---------------|-----------------------------------|
| Generation speed | Slower (HF generate) | 3-5x faster (vLLM) |
| Memory overhead | Lower | Higher (KV cache) |
| Batch generation | Sequential | Continuous batching |
| LoRA rank | Typically 16-32 | Up to 64+ |
| Use case | SFT training | RL training (GRPO/PPO) |

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_RL_Model_Loading]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]

