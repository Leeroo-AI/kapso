# Implementation: save_pretrained_vision

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Computer_Vision]], [[domain::Model_Serialization]], [[domain::Multimodal]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Concrete tool for saving fine-tuned Vision-Language Models with processor (tokenizer + image processor) included.

=== Description ===

VLM saving uses the same `save_pretrained` / `save_pretrained_merged` APIs as text models, but saves the AutoProcessor (which includes both tokenizer and image processor) alongside the model weights.

=== Usage ===

Call after training, passing the processor instead of tokenizer.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L235-860

=== Import ===
<syntaxhighlight lang="python">
# Same API as text models
model.save_pretrained_merged("./output", processor, save_method="merged_16bit")
</syntaxhighlight>

== Usage Examples ==

=== Save VLM with LoRA ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# After training...
# Save LoRA adapters
model.save_pretrained("./vlm_lora")
processor.save_pretrained("./vlm_lora")
</syntaxhighlight>

=== Save Merged VLM ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

# After training...
# Merge and save
model.save_pretrained_merged(
    "./vlm_merged",
    processor,  # Pass processor, not tokenizer
    save_method = "merged_16bit",
)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Unslothai_Unsloth_Vision_Model_Saving]]

=== Requires Environment ===
* [[requires_env::Environment:Unslothai_Unsloth_CUDA_GPU_Environment]]
