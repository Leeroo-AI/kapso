# Implementation: unslothai_unsloth_save_pretrained_lora

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|PEFT|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Concrete tool for saving only LoRA adapter weights without merging with base model.

=== Description ===

`model.save_pretrained()` with LoRA models saves only the adapter weights (~1-2% of full model). This creates:
- `adapter_model.safetensors`: LoRA weights
- `adapter_config.json`: LoRA configuration

Requires the base model to be loaded separately at inference time.

=== Usage ===

Use for lightweight model distribution when users have access to the base model.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/unslothai/unsloth unsloth]
* '''File:''' unsloth/save.py
* '''Lines:''' L100-200

=== Usage Example ===
<syntaxhighlight lang="python">
# Save LoRA adapters only
model.save_pretrained("my_lora_adapter")
tokenizer.save_pretrained("my_lora_adapter")

# Creates:
# my_lora_adapter/
# ├── adapter_model.safetensors  (~100MB for 7B model)
# ├── adapter_config.json
# └── tokenizer files
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:unslothai_unsloth_LoRA_Export]]

=== Requires Environment ===
* [[requires_env::Environment:unslothai_unsloth_CUDA]]
