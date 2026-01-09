# Implementation: get_peft_model_rl

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|GRPO|https://arxiv.org/abs/2402.03300]]
|-
! Domains
| [[domain::NLP]], [[domain::Reinforcement_Learning]], [[domain::Parameter_Efficient_Finetuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for injecting LoRA adapters into a vLLM-enabled model for reinforcement learning training, with rank constraints matching vLLM's pre-allocated limits.

=== Description ===

`FastLanguageModel.get_peft_model` for RL workflows applies LoRA adapters with specific constraints for vLLM compatibility. The LoRA rank must not exceed `max_lora_rank` specified during model loading, as vLLM pre-allocates memory for adapter weights.

This is an angle-specific documentation of `get_peft_model` for the RL LoRA Configuration context.

Key RL-specific considerations:
* **Rank constraint** - `r <= max_lora_rank` from model loading
* **Higher ranks typical** - RL often uses r=32-64 for more capacity
* **vLLM memory** - LoRA weights affect vLLM's available GPU memory

=== Usage ===

Call after loading model with `fast_inference=True`. The LoRA rank must be <= `max_lora_rank` from loading. Higher ranks are common for RL as the model needs more adaptability for policy optimization.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/llama.py
* '''Lines:''' L2577-3100 (same function, RL context)

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def get_peft_model(
    model: PreTrainedModel,  # Model with vLLM engine
    r: int = 16,  # Must be <= max_lora_rank from loading
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_gradient_checkpointing: str = "unsloth",
    use_rslora: bool = False,
    **kwargs,
) -> PeftModelForCausalLM:
    """
    Apply LoRA adapters for RL training.

    Args:
        model: Model with vLLM engine from from_pretrained(fast_inference=True)
        r: LoRA rank (MUST be <= max_lora_rank from loading)
        target_modules: Modules to apply LoRA
        lora_alpha: LoRA scaling factor
        use_rslora: Use rank-stabilized scaling

    Returns:
        PeftModelForCausalLM ready for GRPOTrainer
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
| model || PreTrainedModel || Yes || Model with vLLM engine from fast_inference=True loading
|-
| r || int || No (default: 16) || LoRA rank; MUST be <= max_lora_rank from loading
|-
| target_modules || List[str] || No || Modules for LoRA; defaults to attention + MLP
|-
| lora_alpha || int || No (default: 16) || Scaling factor; effective scale = alpha/r
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModelForCausalLM || vLLM-enabled model with LoRA adapters
|}

== Usage Examples ==

=== Standard RL LoRA Configuration ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with vLLM and max_lora_rank=64
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.5,
)

# Apply LoRA with rank <= 64
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,          # Max rank, equal to max_lora_rank
    lora_alpha = 64,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== Lower Rank for Memory Efficiency ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load with max_lora_rank=64
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 64,
    gpu_memory_utilization = 0.6,
)

# Use lower rank to save memory
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,          # Lower than max for memory
    lora_alpha = 32,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_RL_LoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_vLLM_Environment]]
