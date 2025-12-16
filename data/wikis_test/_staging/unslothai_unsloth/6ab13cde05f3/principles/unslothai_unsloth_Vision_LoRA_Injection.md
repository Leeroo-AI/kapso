{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
* [[source::Doc|Vision Fine-tuning|https://docs.unsloth.ai/basics/vision-fine-tuning]]
|-
! Domains
| [[domain::Vision_Language_Models]], [[domain::LoRA]], [[domain::Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Process of applying LoRA adapters to Vision-Language Models with granular control over which components (vision encoder, language model, attention, MLP) receive adapters.

=== Description ===

Vision LoRA injection extends standard LoRA with VLM-specific considerations:

**Component Selection:**
- Vision encoder layers (often kept frozen or lightly tuned)
- Cross-attention layers (connect vision and language)
- Language model layers (main training target)
- Projection layers (vision-to-text mapping)

**Fine-tuning Strategies:**
1. **Language-only**: Freeze vision, train language (fastest)
2. **Vision-only**: Train vision, freeze language (for visual tasks)
3. **Full VLM**: Train both components (best quality, highest memory)

=== Usage ===

Apply VLM LoRA when:
- Fine-tuning for visual question answering
- Adapting OCR/document understanding models
- Training image captioning systems

== Practical Guide ==

=== Train Both Vision and Language ===
<syntaxhighlight lang="python">
from unsloth import FastVisionModel

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    target_modules="all-linear",
    finetune_vision_layers=True,      # Train vision encoder
    finetune_language_layers=True,    # Train language model
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
)
</syntaxhighlight>

=== Language-Only Training ===
<syntaxhighlight lang="python">
# Faster, less memory, good for text-focused tasks
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    finetune_vision_layers=False,     # Freeze vision
    finetune_language_layers=True,
)
</syntaxhighlight>

=== Vision-Only Training ===
<syntaxhighlight lang="python">
# For improving visual understanding
model = FastVisionModel.get_peft_model(
    model,
    r=32,
    finetune_vision_layers=True,
    finetune_language_layers=False,   # Freeze language
)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_FastVisionModel]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Vision_Language_Model_Finetuning]]
