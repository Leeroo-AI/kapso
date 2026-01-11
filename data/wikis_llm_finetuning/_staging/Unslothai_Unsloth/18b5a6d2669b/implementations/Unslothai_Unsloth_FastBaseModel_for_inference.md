# Implementation: FastBaseModel_for_inference

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Inference]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for configuring Vision-Language Models for efficient inference mode after training.

=== Description ===

`FastVisionModel.for_inference` prepares a VLM for generation by:
* Disabling gradient computation
* Disabling gradient checkpointing
* Setting model to evaluation mode

This reduces memory usage and speeds up generation.

=== Usage ===

Call after training to switch to inference mode for evaluation or deployment.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/models/vision.py
* '''Lines:''' L1191-1250

=== Signature ===
<syntaxhighlight lang="python">
@staticmethod
def for_inference(model: PreTrainedModel) -> None:
    """
    Configure model for inference mode.

    Args:
        model: VLM with LoRA adapters

    Returns:
        None (modifies model in place)
    """
</syntaxhighlight>

== Usage Examples ==

=== Switch to Inference Mode ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# After training...
trainer.train()

# Switch to inference mode
FastVisionModel.for_inference(model)
# Or: model.for_inference()

# Now generate
with torch.no_grad():
    outputs = model.generate(
        **processor(images=image, text=prompt, return_tensors="pt"),
        max_new_tokens=256,
    )
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Inference_Mode]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
