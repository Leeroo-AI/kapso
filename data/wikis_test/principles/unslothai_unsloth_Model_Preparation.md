{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|GGUF Export|https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Deployment]], [[domain::GGUF]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of loading and preparing a trained model for GGUF export, ensuring weights are ready for merging and conversion.

=== Description ===

Model preparation for GGUF export ensures:
- Model is properly loaded with correct configuration
- LoRA adapters are attached if training from saved adapters
- Base model weights are accessible for merging
- Tokenizer is ready for vocabulary export

== Practical Guide ==

=== Load from Training Session ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Model is already loaded and trained
# Proceed directly to export
model.save_pretrained_gguf(...)
</syntaxhighlight>

=== Load Saved LoRA Adapters ===
<syntaxhighlight lang="python">
from unsloth import FastLanguageModel

# Load previously saved LoRA adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # Saved adapter directory
    max_seq_length=2048,
    load_in_4bit=True,
)

# Now export to GGUF
model.save_pretrained_gguf(...)
</syntaxhighlight>

=== Verify Model State ===
<syntaxhighlight lang="python">
# Check if model has LoRA adapters
from peft import PeftModel
if isinstance(model, PeftModel):
    print("Model has LoRA adapters - will merge before export")
else:
    print("Model is base model - will export directly")
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_GGUF_Export]]
