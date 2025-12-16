{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Save Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::Model_Export]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of saving trained Vision-Language Models with both vision and language adapter weights for deployment.

=== Description ===

Vision model saving handles:
- Combined vision and language LoRA weights
- Processor/tokenizer configuration
- Image preprocessing settings
- Model configuration for multimodal inference

== Practical Guide ==

=== Save LoRA Adapters ===
<syntaxhighlight lang="python">
# Save both vision and language adapters
model.save_pretrained("vision_lora_model")
tokenizer.save_pretrained("vision_lora_model")
</syntaxhighlight>

=== Save Merged Model ===
<syntaxhighlight lang="python">
# Merge adapters and save full model
model.save_pretrained_merged(
    "vision_merged_model",
    tokenizer,
    save_method="merged_16bit",
)
</syntaxhighlight>

=== Push to Hub ===
<syntaxhighlight lang="python">
model.push_to_hub_merged(
    "your-username/my-vlm-model",
    tokenizer,
    save_method="merged_16bit",
    token="hf_...",
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Vision_Language_Model_Finetuning]]
