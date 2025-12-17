# Principle: unslothai_unsloth_LoRA_Export

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|PEFT|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Parameter_Efficient]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for exporting only LoRA adapter weights for lightweight model distribution.

=== Description ===

LoRA Export saves trained adapter weights separately from the base model. Benefits:
1. **Small file size**: ~1-2% of full model
2. **Quick updates**: Easy to iterate on adapters
3. **Multiple adapters**: Can share one base model with many adapters

Drawbacks:
- Requires base model at load time
- Users need PEFT library

=== Usage ===

Use when distributing fine-tunes where users have access to the base model.

== File Structure ==

<syntaxhighlight lang="python">
# LoRA adapter files
adapter_files = {
    "adapter_model.safetensors": "LoRA weight matrices (A, B)",
    "adapter_config.json": {
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj", ...],
        "base_model_name_or_path": "meta-llama/...",
    }
}
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_lora]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
