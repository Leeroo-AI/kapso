{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::File|config.py|src/peft/tuners/lora/config.py]]
|-
! Domains
| [[domain::Fine_Tuning]], [[domain::Quantization]], [[domain::Memory_Optimization]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for creating LoraConfig optimized for QLoRA training on quantized models.

=== Description ===

This is the same `LoraConfig` class used for standard LoRA, but with parameters tuned for quantized training. The key recommendation is using `target_modules="all-linear"` to adapt all linear layers, maximizing the benefit of QLoRA's memory efficiency.

=== Usage ===

Create LoraConfig with QLoRA-optimized parameters before calling `get_peft_model()` on a quantized model.

== Code Reference ==

=== Source Location ===
* '''File:''' `src/peft/tuners/lora/config.py`
* '''Lines:''' L47-300

=== Signature ===
<syntaxhighlight lang="python">
@dataclass
class LoraConfig(PeftConfig):
    r: int = 8
    lora_alpha: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Optional[TaskType] = None
    # ... additional parameters
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import LoraConfig, TaskType
</syntaxhighlight>

== Usage Examples ==

=== QLoRA Config for Causal LM ===
<syntaxhighlight lang="python">
from peft import LoraConfig, TaskType

qlora_config = LoraConfig(
    r=16,  # Slightly higher rank for QLoRA
    lora_alpha=32,
    target_modules="all-linear",  # Adapt all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
</syntaxhighlight>

=== QLoRA with Specific Modules ===
<syntaxhighlight lang="python">
qlora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_QLoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
