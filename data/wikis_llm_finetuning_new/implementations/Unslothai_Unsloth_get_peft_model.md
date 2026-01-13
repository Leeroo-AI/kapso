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
| [[domain::NLP]], [[domain::Parameter_Efficient_Training]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for injecting LoRA adapters into language models provided by the Unsloth library.

=== Description ===

`FastLanguageModel.get_peft_model` wraps the PEFT library's LoRA injection with Unsloth-specific optimizations. It applies trainable low-rank adapter matrices to specified target modules while keeping the base model frozen.

Key features:
* Automatic target module selection for common architectures
* Unsloth's optimized gradient checkpointing integration
* Support for RSLoRA (Rank-Stabilized LoRA) and LoftQ initialization
* Memory-efficient handling of embedding and lm_head training
* Validation of existing adapters to prevent conflicts

=== Usage ===

Import this function after loading a model with `FastLanguageModel.from_pretrained`. Use it to add trainable LoRA adapters before starting training. This is required for QLoRA workflows where the base model is quantized.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/llama.py
* '''Lines:''' 2630-3142

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
    use_gradient_checkpointing: Union[str, bool] = "unsloth",
    random_state: int = 3407,
    max_seq_length: int = 2048,
    use_rslora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    init_lora_weights: Union[bool, str] = True,
    loftq_config: dict = {},
    temporary_location: str = "_unsloth_temporary_saved_buffers",
    qat_scheme: Optional[str] = None,
    ensure_weight_tying: bool = False,
    **kwargs,
) -> PeftModelForCausalLM:
    """
    Add LoRA adapters to a pre-trained model.

    Args:
        model: Pre-trained model from from_pretrained
        r: LoRA rank (higher = more parameters, more capacity)
        target_modules: List of module names to apply LoRA
        lora_alpha: LoRA scaling factor (effective scale = alpha/r)
        lora_dropout: Dropout rate for LoRA layers (0 recommended for speed)
        bias: Bias training mode ("none", "all", "lora_only")
        layers_to_transform: Specific layer indices to apply LoRA
        use_gradient_checkpointing: Memory optimization mode
        use_rslora: Use Rank-Stabilized LoRA scaling
        modules_to_save: Additional modules to unfreeze and train
        init_lora_weights: Initialization method (True, False, "gaussian", "loftq")

    Returns:
        PeftModelForCausalLM with LoRA adapters ready for training
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# get_peft_model is a method on FastLanguageModel
model = FastLanguageModel.get_peft_model(model, r=16, ...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Base model from FastLanguageModel.from_pretrained
|-
| r || int || No || LoRA rank (default: 16, higher = more capacity)
|-
| target_modules || List[str] || No || Module names to apply LoRA (default: all attention + MLP projections)
|-
| lora_alpha || int || No || Scaling factor (default: 16, effective scale = alpha/r)
|-
| lora_dropout || float || No || Dropout rate (default: 0.0, 0 recommended for Unsloth speed)
|-
| bias || str || No || Bias training mode (default: "none")
|-
| use_gradient_checkpointing || str/bool || No || Memory optimization (default: "unsloth")
|-
| use_rslora || bool || No || Use Rank-Stabilized LoRA (default: False)
|-
| modules_to_save || List[str] || No || Additional modules to train (e.g., ["embed_tokens", "lm_head"])
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModelForCausalLM || Model with LoRA adapters injected, ready for training
|}

== Usage Examples ==

=== Basic LoRA Injection ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load model first
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters with default settings
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Check trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
</syntaxhighlight>

=== Higher Rank for Complex Tasks ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Higher rank for more complex adaptation
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # Higher rank = more trainable parameters
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    use_rslora=True,  # Rank-stabilized scaling
)
</syntaxhighlight>

=== Training Embeddings ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Include embeddings and lm_head for vocabulary adaptation
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    modules_to_save=["embed_tokens", "lm_head"],  # Also train these
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_LoRA_Adapter_Injection]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_PEFT]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_LoRA_Rank_Selection]]
* [[uses_heuristic::Heuristic:Unslothai_Unsloth_Gradient_Checkpointing]]
