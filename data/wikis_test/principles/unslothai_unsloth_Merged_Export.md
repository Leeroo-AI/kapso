# Principle: unslothai_unsloth_Merged_Export

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Unsloth Docs|https://docs.unsloth.ai]]
* [[source::Doc|SafeTensors|https://huggingface.co/docs/safetensors]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Export]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for exporting models with LoRA weights merged into base weights as standalone models.

=== Description ===

Merged Export produces a standalone model by:
1. Dequantizing 4-bit base weights to 16-bit
2. Computing LoRA contribution: W' = W + (α/r)BA
3. Saving merged weights in safetensors format

Result is a standard HuggingFace model that works without PEFT or Unsloth.

=== Usage ===

Use for deployment where you want a single, dependency-free model.

== Merge Mathematics ==

<math>
W_{merged} = W_{base} + \frac{\alpha}{r} \cdot B \cdot A
</math>

The merged model behaves identically to the LoRA model but loads as a standard model.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
