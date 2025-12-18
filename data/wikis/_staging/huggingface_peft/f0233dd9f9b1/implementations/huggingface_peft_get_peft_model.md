{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Docs|https://huggingface.co/docs/peft/quicktour]]
|-
! Domains
| [[domain::NLP]], [[domain::Fine_Tuning]], [[domain::Adapter]]
|-
! Last Updated
| [[last_updated::2025-01-15 12:00 GMT]]
|}

== Overview ==

Concrete tool for wrapping a pre-trained model with PEFT adapter layers for parameter-efficient fine-tuning.

=== Description ===

`get_peft_model` is the primary factory function that transforms a base transformer model into a PEFT-enabled model. It injects adapter layers (LoRA, etc.) based on the provided configuration, freezes the base model weights, and sets up the model for efficient fine-tuning where only the adapter parameters are trained.

=== Usage ===

Call this function after loading your base model and creating a `LoraConfig`. The function modifies the model in-place and returns a `PeftModel` wrapper. After calling this, use `model.print_trainable_parameters()` to verify that only adapter weights are trainable.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/huggingface/peft peft]
* '''File:''' src/peft/mapping_func.py
* '''Lines:''' L30-128

=== Signature ===
<syntaxhighlight lang="python">
def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config.

    The model is modified in-place with adapter layers injected.

    Args:
        model: Base transformer model to wrap
        peft_config: Configuration object (LoraConfig, etc.)
        adapter_name: Name for the adapter (default: "default")
        mixed: Allow mixing different adapter types
        autocast_adapter_dtype: Auto-cast adapter weights to float32
        revision: Base model revision for saving
        low_cpu_mem_usage: Create empty weights on meta device

    Returns:
        PeftModel: Model with adapter layers ready for training
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from peft import get_peft_model
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Base transformer model to adapt
|-
| peft_config || PeftConfig || Yes || LoraConfig or other PEFT configuration
|-
| adapter_name || str || No || Identifier for the adapter. Default: "default"
|-
| mixed || bool || No || Allow mixing adapter types (LoRA + IA3). Default: False
|-
| autocast_adapter_dtype || bool || No || Cast float16/bfloat16 adapters to float32. Default: True
|-
| low_cpu_mem_usage || bool || No || Initialize empty weights on meta device. Default: False
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| peft_model || PeftModel || Wrapped model with adapter layers injected and base weights frozen
|}

== Usage Examples ==

=== Standard LoRA Model Creation ===
<syntaxhighlight lang="python">
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2. Configure LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# 3. Create PEFT model
model = get_peft_model(model, config)

# 4. Verify trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
</syntaxhighlight>

=== With Named Adapter ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig

# Create model with named adapter
model = get_peft_model(
    model,
    config,
    adapter_name="math_adapter",  # Custom name for later reference
)

# Access adapter by name
print(model.active_adapter)  # "math_adapter"
</syntaxhighlight>

=== Low Memory Mode ===
<syntaxhighlight lang="python">
from peft import get_peft_model, LoraConfig

# For large models, use low_cpu_mem_usage
model = get_peft_model(
    model,
    config,
    low_cpu_mem_usage=True,  # Defer weight materialization
)
# Weights will be loaded when adapter is loaded via load_adapter()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:huggingface_peft_PEFT_Model_Creation]]

=== Requires Environment ===
* [[requires_env::Environment:huggingface_peft_Core_Environment]]
