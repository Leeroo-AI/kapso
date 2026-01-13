# Principle: Model_Saving

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|HuggingFace Model Saving|https://huggingface.co/docs/transformers/main_classes/model]]
* [[source::Doc|PEFT Saving|https://huggingface.co/docs/peft/tutorial/peft_model_config]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::NLP]], [[domain::Model_Serialization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for persisting trained model weights in various formats for deployment, sharing, or continued training.

=== Description ===

Model Saving in the context of QLoRA involves converting the trained model weights to a persistent format. There are three primary save methods:

1. **LoRA Adapters Only**: Save only the small adapter weights (~10-100MB). The base model must be available at inference time. Best for sharing and storage efficiency.

2. **Merged 16-bit**: Dequantize the 4-bit base model and merge LoRA weights into full 16-bit precision. Creates a standalone model (~2-14GB per billion parameters). Required for GGUF conversion.

3. **Merged 4-bit**: Merge LoRA weights then re-quantize to 4-bit. Smaller than 16-bit but requires bitsandbytes for inference.

The merging process involves dequantizing NF4 weights to float16, applying the LoRA delta (W' = W + BA), and optionally re-quantizing.

=== Usage ===

Use this principle when:
* Persisting a trained model for later inference
* Sharing a fine-tuned model on HuggingFace Hub
* Preparing a model for GGUF conversion (Ollama, llama.cpp)
* Creating checkpoints during training

This is typically the final step in a fine-tuning workflow.

== Theoretical Basis ==

'''LoRA Merging Formula:'''
<math>
W_{merged} = W_{base} + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
- W_base is the (dequantized) base model weight
- B, A are the trained LoRA matrices
- α is lora_alpha, r is the rank
- The scaling factor α/r controls the adaptation strength

'''Save Methods Comparison:'''
{| class="wikitable"
|-
! Method !! Size !! Inference Requirement !! Use Case
|-
| lora || ~10-100MB || Base model + PEFT || Training checkpoints, sharing adapters
|-
| merged_16bit || ~2GB/B params || None (standalone) || GGUF conversion, production
|-
| merged_4bit || ~0.5GB/B params || bitsandbytes || Memory-constrained inference
|}

'''Pseudo-code for Merging:'''
<syntaxhighlight lang="python">
# Abstract algorithm for LoRA merge
for layer in model.layers:
    if has_lora_adapter(layer):
        W_base = dequantize(layer.weight)  # NF4 -> float16
        A, B = get_lora_matrices(layer)
        scale = lora_alpha / lora_rank
        W_merged = W_base + scale * (B @ A)
        layer.weight = W_merged  # Now float16
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_save_pretrained_merged]]

