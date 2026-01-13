# Implementation: get_peft_model_vision

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Parameter_Efficient_Training]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Concrete tool for injecting LoRA adapters into vision-language models with component-specific control provided by Unsloth.

=== Description ===

`FastVisionModel.get_peft_model` extends standard LoRA injection with vision-specific parameters:

* `finetune_vision_layers`: Apply LoRA to vision encoder
* `finetune_language_layers`: Apply LoRA to language model
* `finetune_attention_modules`: Apply LoRA to attention layers
* `finetune_mlp_modules`: Apply LoRA to MLP layers

The function automatically determines the correct target modules for each VLM architecture using regex patterns.

=== Usage ===

Call this method on a vision model after loading with `FastVisionModel.from_pretrained`. Configure the component flags based on your task requirements and memory constraints.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/vision.py
* '''Lines:''' 920-1076 (FastBaseModel.get_peft_model)

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def get_peft_model(
    model: PreTrainedModel,
    r: int = 16,
    target_modules: Optional[Union[str, List[str]]] = None,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
    # Vision-specific flags
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    # Standard parameters
    layers_to_transform: Optional[List[int]] = None,
    layers_pattern: Optional[str] = None,
    use_gradient_checkpointing: Union[str, bool] = "unsloth",
    random_state: int = 3407,
    use_rslora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    init_lora_weights: Union[bool, str] = True,
    loftq_config: dict = {},
    **kwargs,
) -> PeftModelForCausalLM:
    """
    Add LoRA adapters to a vision-language model.

    Args:
        model: Vision model from FastVisionModel.from_pretrained
        r: LoRA rank
        target_modules: Override automatic module selection (or "all-linear")
        lora_alpha: LoRA scaling factor
        finetune_vision_layers: Apply LoRA to vision encoder
        finetune_language_layers: Apply LoRA to language model
        finetune_attention_modules: Apply LoRA to attention layers
        finetune_mlp_modules: Apply LoRA to MLP layers
        use_gradient_checkpointing: Memory optimization mode

    Returns:
        PeftModelForCausalLM with vision-aware LoRA adapters
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# get_peft_model is a method on FastVisionModel
model = FastVisionModel.get_peft_model(model, r=16, ...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || Vision model from FastVisionModel.from_pretrained
|-
| r || int || No || LoRA rank (default: 16)
|-
| target_modules || str/List[str] || No || Override module selection (default: auto-determined)
|-
| lora_alpha || int || No || LoRA scaling factor (default: 16)
|-
| finetune_vision_layers || bool || No || Apply LoRA to vision encoder (default: True)
|-
| finetune_language_layers || bool || No || Apply LoRA to language model (default: True)
|-
| finetune_attention_modules || bool || No || Apply LoRA to attention (default: True)
|-
| finetune_mlp_modules || bool || No || Apply LoRA to MLPs (default: True)
|-
| use_gradient_checkpointing || str/bool || No || Memory optimization (default: "unsloth")
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModelForCausalLM || Model with vision-aware LoRA adapters
|}

== Usage Examples ==

=== Full Vision-Language LoRA ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Train both vision and language components
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    use_gradient_checkpointing="unsloth",
)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,}")
</syntaxhighlight>

=== Language-Only LoRA ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Only train language model, keep vision encoder frozen
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    finetune_vision_layers=False,  # Freeze vision encoder
    finetune_language_layers=True,
    use_gradient_checkpointing="unsloth",
)

# Fewer trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable (language only): {trainable:,}")
</syntaxhighlight>

=== Attention-Only LoRA ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Only train attention modules (both vision and language)
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=False,  # Skip MLP layers
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

=== Using "all-linear" Target ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA to all linear layers (ignores finetune_* flags)
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules="all-linear",  # Override automatic selection
    use_gradient_checkpointing="unsloth",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_LoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_PEFT]]

