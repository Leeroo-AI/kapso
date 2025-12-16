{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Save Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
|-
! Domains
| [[domain::Reinforcement_Learning]], [[domain::Model_Export]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of saving RL-trained models with options for LoRA adapters or merged weights.

=== Description ===

RL model saving is identical to SFT saving:
- LoRA adapters only (~50-200MB)
- Merged 16-bit (multi-GB, standalone)
- GGUF for local deployment

== Practical Guide ==

=== Save LoRA Adapters ===
<syntaxhighlight lang="python">
# Fastest, smallest
model.save_pretrained("grpo_lora_model")
tokenizer.save_pretrained("grpo_lora_model")
</syntaxhighlight>

=== Save Merged Model ===
<syntaxhighlight lang="python">
model.save_pretrained_merged(
    "grpo_merged_model",
    tokenizer,
    save_method="merged_16bit",
)
</syntaxhighlight>

=== Push to Hub ===
<syntaxhighlight lang="python">
model.push_to_hub_merged(
    "your-username/model-grpo",
    tokenizer,
    save_method="merged_16bit",
    token="hf_...",
)
</syntaxhighlight>

=== Save to GGUF ===
<syntaxhighlight lang="python">
model.save_pretrained_gguf(
    "grpo_gguf_model",
    tokenizer,
    quantization_method="q4_k_m",
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GRPO_Reinforcement_Learning]]
