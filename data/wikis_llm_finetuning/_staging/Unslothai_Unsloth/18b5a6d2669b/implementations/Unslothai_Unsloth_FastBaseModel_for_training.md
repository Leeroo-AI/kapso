# Implementation: FastBaseModel_for_training

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Training]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for configuring Vision-Language Models for training mode with gradient checkpointing and proper gradient computation enabled.

=== Description ===

`FastVisionModel.for_training` (implemented in `FastBaseModel`) prepares a VLM for fine-tuning by:
* Enabling gradient computation
* Activating gradient checkpointing
* Setting model to training mode

This is necessary before passing the model to SFTTrainer.

=== Usage ===

Call before starting training if the model was previously in inference mode. Typically called automatically by Unsloth, but can be called explicitly.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/vision.py
* '''Lines:''' L1190-1250

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def for_training(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True,
) -> None:
    """
    Configure model for training mode.

    Args:
        model: VLM with LoRA adapters
        use_gradient_checkpointing: Enable gradient checkpointing

    Returns:
        None (modifies model in place)
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel
# Called as: FastVisionModel.for_training(model) or model.for_training()
</syntaxhighlight>

== Usage Examples ==

=== Explicit Training Mode ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(...)
model = FastVisionModel.get_peft_model(model, ...)

# Explicitly enable training mode
FastVisionModel.for_training(model)

# Or use the attached method
model.for_training()
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Training_Mode]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
