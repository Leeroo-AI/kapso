# Principle: Model_Saving

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|Safetensors|https://huggingface.co/docs/safetensors]]
* [[source::Doc|PEFT Saving|https://huggingface.co/docs/peft/quicktour#save-and-load-a-model]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Serialization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Techniques for serializing fine-tuned model weights to disk in formats suitable for different deployment scenarios and downstream processing.

=== Description ===

Model Saving after LoRA fine-tuning involves choosing between three fundamental strategies:

1. **Adapter-only saving**: Store just the LoRA matrices (A, B weights). Requires base model for inference but is fastest and smallest.

2. **Merged saving**: Combine LoRA weights with base model (W' = W₀ + BA). Creates standalone model but requires dequantization if base was quantized.

3. **Format conversion**: Transform merged weights into deployment formats (GGUF for llama.cpp, ONNX for production).

The choice impacts inference requirements, model size, and downstream compatibility.

=== Usage ===

Choose saving strategy based on deployment target:

| Strategy | Size | Speed | Use Case |
|----------|------|-------|----------|
| LoRA only | ~100MB | Fast | HuggingFace inference, continued training |
| Merged 16-bit | ~14GB (7B) | Slow | GGUF export, framework conversion |
| Merged 4-bit | ~4GB (7B) | Medium | Direct quantized inference |

== Theoretical Basis ==

=== LoRA Merging ===

For a layer with LoRA adapters:

<math>
W' = W_0 + \frac{\alpha}{r} BA
</math>

Merging computes W' explicitly, eliminating the separate forward pass through LoRA matrices.

If W₀ is quantized, merging requires:
1. Dequantize W₀ to float16/float32
2. Compute W' = W₀ + (α/r)BA
3. Save W' in target precision

=== Sharding Strategy ===

Large models are split into shards for efficient loading:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
# Model sharding (abstract)
def save_sharded(state_dict, max_shard_size):
    shards = []
    current_shard = {}
    current_size = 0

    for name, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    return shards  # model-00001-of-00003.safetensors, etc.
</syntaxhighlight>

=== Safetensors Format ===

Modern format replacing pickle-based `.bin` files:
* Memory-mapped loading (lazy load)
* No arbitrary code execution (security)
* Parallel loading across shards
* Header contains tensor metadata

=== Hub Integration ===

Saving to HuggingFace Hub enables:
* Version control (git-based)
* Model cards (documentation)
* Inference API endpoints
* Community sharing

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_save_pretrained]]
