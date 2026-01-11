# Implementation: get_peft_model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::NLP]], [[domain::Parameter_Efficient_Finetuning]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for injecting LoRA (Low-Rank Adaptation) adapters into a quantized language model for parameter-efficient fine-tuning, provided by Unsloth's optimized PEFT wrapper.

=== Description ===

`FastLanguageModel.get_peft_model` wraps PEFT's LoRA implementation with Unsloth-specific optimizations. It attaches trainable low-rank matrices to specified attention and MLP modules while keeping base weights frozen. The function applies fused LoRA kernels for QKV projections and handles gradient checkpointing setup.

Key features for QLoRA training:
* **Fused LoRA operations** - Single kernel for QKV attention projections
* **Optimized gradient checkpointing** - Unsloth's memory-efficient implementation
* **Automatic target module detection** - Defaults to all attention + MLP projections
* **Support for embed_tokens/lm_head training** - Via `modules_to_save` parameter

=== Usage ===

Call this function immediately after loading a model with `FastLanguageModel.from_pretrained`. Pass the model and configure LoRA hyperparameters (rank, alpha, target modules). The returned model has LoRA adapters attached and is ready for training.

Typical LoRA rank values:
* r=8-16: Standard fine-tuning tasks
* r=32-64: Complex tasks requiring more capacity
* r=128+: Full fine-tuning approximation (rare)

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/llama.py
* '''Lines:''' L2577-3100

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def get_peft_model(
    model: PreTrainedModel,
    r: int = 16,
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
    layers_to_transform: Optional[List[int]] = None,
    layers_pattern: Optional[str] = None,
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 3407,
    max_seq_length: int = 2048,
    use_rslora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    init_lora_weights: bool = True,
    loftq_config: dict = {},
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    qat_scheme: Optional[str] = None,
    ensure_weight_tying: bool = False,
    **kwargs,
) -> PeftModelForCausalLM:
    """
    Apply LoRA adapters to a pretrained model.

    Args:
        model: Base model from FastLanguageModel.from_pretrained
        r: LoRA rank (dimensionality of low-rank matrices)
        target_modules: List of module names to apply LoRA
        lora_alpha: LoRA scaling factor (effective scale = alpha/r)
        lora_dropout: Dropout probability (0.0 recommended for Unsloth)
        bias: Bias training mode ("none", "all", "lora_only")
        use_gradient_checkpointing: "unsloth" for optimized checkpointing
        use_rslora: Enable rank-stabilized LoRA scaling
        modules_to_save: Modules to train fully (e.g., embed_tokens, lm_head)

    Returns:
        PeftModelForCausalLM with LoRA adapters attached
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel
# Called as: FastLanguageModel.get_peft_model(model, ...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Base model from FastLanguageModel.from_pretrained
|-
| r || int || No (default: 16) || LoRA rank - dimensionality of adaptation matrices
|-
| target_modules || List[str] || No || Modules to apply LoRA; defaults to all attention + MLP
|-
| lora_alpha || int || No (default: 16) || LoRA scaling factor; effective scale = alpha/r
|-
| lora_dropout || float || No (default: 0.0) || Dropout rate; 0.0 recommended for fast patching
|-
| use_rslora || bool || No (default: False) || Use rank-stabilized LoRA (scale = alpha/sqrt(r))
|-
| modules_to_save || List[str] || No || Full-precision trainable modules (embed_tokens, lm_head)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModelForCausalLM || Model with LoRA adapters; only adapter weights are trainable
|}

== Usage Examples ==

=== Standard QLoRA Configuration ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                # LoRA rank
    lora_alpha = 16,       # Scaling factor (effective scale = 1.0)
    lora_dropout = 0,      # No dropout for fast patching
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 1,249,902,592 || trainable%: 1.0906%
</syntaxhighlight>

=== Training Embeddings (Extended Vocabulary) ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Train LoRA + embedding layers for new tokens
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    modules_to_save = ["embed_tokens", "lm_head"],  # Train embeddings
    use_gradient_checkpointing = "unsloth",
)
</syntaxhighlight>

=== High-Rank Configuration ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Higher rank for complex tasks
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,                # Higher rank for more capacity
    lora_alpha = 64,       # Keep alpha = r for scale = 1.0
    use_rslora = True,     # Rank-stabilized scaling
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_LoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection_Tip]]
