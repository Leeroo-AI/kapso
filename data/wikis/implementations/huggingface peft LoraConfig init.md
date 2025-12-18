{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/conceptual_guides/lora]]
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::NLP]], [[domain::Fine_Tuning]], [[domain::Configuration]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for configuring Low-Rank Adaptation parameters for standard LoRA fine-tuning workflows.

=== Description ===

`LoraConfig` is a dataclass that defines all hyperparameters for LoRA adaptation. It specifies which modules to adapt, the rank of the low-rank matrices, scaling factors, and initialization strategies. This configuration is passed to `get_peft_model()` to inject adapter layers into the base model.

=== Usage ===

Use this after loading your base model and before calling `get_peft_model()`. Configure `r` (rank) based on your task complexity - lower ranks (4-8) for simple tasks, higher (16-64) for complex ones. Use `target_modules="all-linear"` to adapt all linear layers, or specify explicit module names for fine-grained control.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/tuners/lora/config.py
* '''Lines:''' L321-879

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class LoraConfig(PeftConfig):
    """
    Configuration class for LoRA adaptation.

    Args:
        r: LoRA attention dimension (rank). Default: 8
        lora_alpha: Scaling factor for LoRA. Default: 8
        target_modules: Modules to apply LoRA. Can be list or "all-linear"
        lora_dropout: Dropout probability for LoRA layers. Default: 0.0
        bias: Bias type - "none", "all", or "lora_only". Default: "none"
        task_type: Task type for model (CAUSAL_LM, SEQ_CLS, etc.)
        use_rslora: Use Rank-Stabilized LoRA scaling. Default: False
        use_dora: Enable Weight-Decomposed LoRA (DoRA). Default: False
        init_lora_weights: Weight initialization strategy. Default: True
        modules_to_save: Additional modules to train. Default: None
    """
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(default=None)
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(default=False)
    bias: Literal["none", "all", "lora_only"] = field(default="none")
    use_rslora: bool = field(default=False)
    modules_to_save: Optional[list[str]] = field(default=None)
    init_lora_weights: bool | Literal["gaussian", "eva", "olora", "pissa", "corda", "loftq"] = field(default=True)
    use_dora: bool = field(default=False)
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import LoraConfig, TaskType
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| r || int || No || LoRA rank dimension. Higher = more capacity but more parameters. Default: 8
|-
| lora_alpha || int || No || Scaling factor. Effective scale is alpha/r. Default: 8
|-
| target_modules || list[str] or str || No || Modules to adapt. "all-linear" for all linear layers, or explicit names
|-
| lora_dropout || float || No || Dropout probability (0.0-0.1 typical). Default: 0.0
|-
| bias || str || No || Bias handling: "none", "all", or "lora_only". Default: "none"
|-
| task_type || TaskType || No || Model task type for proper output layer handling
|-
| use_rslora || bool || No || Enable Rank-Stabilized LoRA (better at low ranks). Default: False
|-
| use_dora || bool || No || Enable DoRA (weight decomposition). Default: False
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| config || LoraConfig || Configuration object ready for get_peft_model()
|}

== Usage Examples ==

=== Standard LoRA Configuration ===
<syntaxhighlight lang="python">
from peft import LoraConfig, TaskType

# Standard LoRA config for causal LM fine-tuning
config = LoraConfig(
    r=16,                          # Rank dimension
    lora_alpha=32,                 # Scaling factor (alpha/r = 2)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,             # Small dropout for regularization
    bias="none",                   # Don't train biases
    task_type=TaskType.CAUSAL_LM,  # For decoder-only models
)
</syntaxhighlight>

=== All-Linear with DoRA ===
<syntaxhighlight lang="python">
from peft import LoraConfig, TaskType

# DoRA config targeting all linear layers
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",   # Adapt all linear layers
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,                 # Enable DoRA for better performance
)
</syntaxhighlight>

=== With Rank-Stabilized LoRA ===
<syntaxhighlight lang="python">
from peft import LoraConfig, TaskType

# RSLoRA for low-rank configurations
config = LoraConfig(
    r=4,                           # Very low rank
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    use_rslora=True,               # Rank-stabilized scaling: alpha/sqrt(r)
    task_type=TaskType.CAUSAL_LM,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_LoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]

=== Uses Heuristic ===
* [[uses_heuristic::Heuristic:huggingface_peft_LoRA_Rank_Selection]]
