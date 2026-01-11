# Implementation: FastBaseModel_get_peft_model

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::NLP]], [[domain::Parameter_Efficient_Finetuning]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for injecting LoRA adapters into Vision-Language Models with fine-grained control over which components (vision, language, attention, MLP) receive adapters.

=== Description ===

`FastVisionModel.get_peft_model` (implemented in `FastBaseModel`) applies LoRA adapters to VLMs with additional parameters for controlling:
* Vision encoder LoRA (`finetune_vision_layers`)
* Language model LoRA (`finetune_language_layers`)
* Attention module targeting (`finetune_attention_modules`)
* MLP module targeting (`finetune_mlp_modules`)

This fine-grained control is essential because different tasks may benefit from different adapter placements.

=== Usage ===

Call after loading VLM with `FastVisionModel.from_pretrained`. Configure which components to adapt based on your task:
* OCR/document: Focus on language layers
* Image understanding: Include vision layers
* General VQA: Both vision and language

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/vision.py
* '''Lines:''' L921-1076

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def get_peft_model(
    model: PreTrainedModel,
    r: int = 16,
    target_modules: Optional[Union[List[str], str]] = None,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    layers_to_transform: Optional[List[int]] = None,
    layers_pattern: Optional[str] = None,
    use_gradient_checkpointing: str = "unsloth",
    random_state: int = 3407,
    use_rslora: bool = False,
    modules_to_save: Optional[List[str]] = None,
    **kwargs,
) -> PeftModelForCausalLM:
    """
    Apply LoRA adapters to a Vision-Language Model.

    Args:
        model: VLM from FastVisionModel.from_pretrained
        r: LoRA rank
        finetune_vision_layers: Apply LoRA to vision encoder
        finetune_language_layers: Apply LoRA to language model
        finetune_attention_modules: Target attention projections
        finetune_mlp_modules: Target MLP projections
        target_modules: Override auto-detection with specific modules
        use_gradient_checkpointing: "unsloth" for memory efficiency

    Returns:
        PeftModelForCausalLM with LoRA adapters
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
# Called as: FastVisionModel.get_peft_model(model, ...)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || PreTrainedModel || Yes || VLM from FastVisionModel.from_pretrained
|-
| r || int || No (default: 16) || LoRA rank
|-
| finetune_vision_layers || bool || No (default: True) || Apply LoRA to vision encoder
|-
| finetune_language_layers || bool || No (default: True) || Apply LoRA to language model
|-
| finetune_attention_modules || bool || No (default: True) || Target attention projections
|-
| finetune_mlp_modules || bool || No (default: True) || Target MLP projections
|-
| target_modules || List[str] or str || No || Override with specific module names
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| model || PeftModelForCausalLM || VLM with LoRA adapters attached
|}

== Usage Examples ==

=== Full VLM LoRA (Vision + Language) ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    load_in_4bit = True,
)

# Apply LoRA to both vision and language
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    finetune_vision_layers = True,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    use_gradient_checkpointing = "unsloth",
)

model.print_trainable_parameters()
</syntaxhighlight>

=== Language-Only LoRA (OCR/Document) ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    load_in_4bit = True,
)

# Only adapt language model for text-heavy tasks
model = FastVisionModel.get_peft_model(
    model,
    r = 16,
    finetune_vision_layers = False,   # Keep vision frozen
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
)
</syntaxhighlight>

=== Attention-Only LoRA ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True,
)

# Only attention, not MLP (fewer parameters)
model = FastVisionModel.get_peft_model(
    model,
    r = 32,
    finetune_vision_layers = True,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = False,  # Skip MLP
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_LoRA_Configuration]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
