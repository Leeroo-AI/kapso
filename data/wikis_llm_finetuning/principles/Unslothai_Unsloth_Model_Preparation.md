# Principle: Model_Preparation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Model Saving|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Doc|PEFT Merging|https://huggingface.co/docs/peft/conceptual_guides/lora]]
|-
! Domains
| [[domain::Model_Serialization]], [[domain::GGUF]], [[domain::LoRA]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Preparing fine-tuned models with LoRA adapters for GGUF conversion by merging adapters into base weights.

=== Description ===

Model Preparation for GGUF export requires merging LoRA adapters into the base model weights:

1. **LoRA Merging**: Combine adapter weights with base model
2. **Precision Selection**: Choose float16 or bfloat16 for merged weights
3. **Memory Management**: Handle large model dequantization for 4-bit models
4. **Tokenizer Alignment**: Ensure tokenizer is saved alongside model

=== Usage ===

Use Model Preparation before GGUF export. The merged model serves as input to the quantization pipeline.

== Theoretical Basis ==

=== LoRA Merging Formula ===

For each LoRA layer, merged weight:

<math>
W_{merged} = W_{base} + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
* W_base: Original base model weight
* B, A: LoRA decomposition matrices
* α: LoRA alpha (scaling factor)
* r: LoRA rank

=== 4-bit Dequantization ===

For QLoRA models:

1. Dequantize 4-bit base weights to float16
2. Merge LoRA weights
3. Save as standard float16 safetensors

Memory requirement peaks during dequantization:

<math>
\text{Peak Memory} \approx 2 \times \text{Model Size (4-bit)} + \text{16-bit layer}
</math>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_unsloth_save_model_merged]]

