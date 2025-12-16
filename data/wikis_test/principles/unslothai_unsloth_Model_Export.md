{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|Safetensors Format|https://huggingface.co/docs/safetensors]]
* [[source::Doc|HuggingFace Model Hub|https://huggingface.co/docs/hub/models]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Deployment]], [[domain::Serialization]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Process of merging trained LoRA adapter weights back into base model weights and serializing the result to HuggingFace-compatible format for deployment or distribution.

=== Description ===
Model export with LoRA merging converts a fine-tuned model (base + adapters) into a standalone model that can be deployed without the PEFT library. The merge operation mathematically combines the low-rank updates into the original weight matrices.

The export process handles:
1. **LoRA Merging**: Fuse A and B matrices into base weights: W_new = W_old + scale * (B @ A)
2. **Dequantization**: Convert 4-bit quantized weights to FP16/BF16 for saving
3. **Memory Management**: Process weights layer-by-layer to minimize peak memory usage
4. **Serialization**: Save in safetensors format with proper sharding for large models
5. **Tokenizer Export**: Include tokenizer configuration for complete model package

Export modes include:
- **merged_16bit**: Full merge to float16 (standard deployment format)
- **merged_4bit**: Keep merged weights quantized (for memory-constrained inference)
- **lora**: Save only adapter weights (smallest files, requires base model at inference)

=== Usage ===
Use model export when:
- Deploying a fine-tuned model to production
- Sharing a model on HuggingFace Hub
- Converting for inference frameworks (vLLM, TensorRT-LLM, SGLang)
- Creating a checkpoint for further training or GGUF conversion

Choose merged_16bit when:
- Deploying to systems without PEFT/BitsAndBytes
- Maximum inference speed is required
- Converting to other formats (GGUF, ONNX)

Choose lora when:
- Storage space is limited
- Multiple adapters share a base model
- You need to hot-swap between fine-tuned variants

== Theoretical Basis ==
LoRA merge combines the low-rank update with the original weight:

<math>
W_{merged} = W_0 + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
- <math>W_0 \in \mathbb{R}^{d \times k}</math>: Original frozen weights
- <math>B \in \mathbb{R}^{d \times r}</math>: Trained down-projection
- <math>A \in \mathbb{R}^{r \times k}</math>: Trained up-projection
- <math>\alpha</math>: LoRA scaling factor
- <math>r</math>: LoRA rank

'''Merge Process for 4-bit Models:'''
<syntaxhighlight lang="python">
# Pseudo-code for 4-bit LoRA merge
def merge_lora_4bit(W0_quantized, A, B, alpha, r, scales):
    """
    Merge LoRA into 4-bit quantized base weights.
    Must dequantize, merge, then optionally requantize.
    """
    # Step 1: Dequantize base weights to FP16
    W0_fp16 = dequantize_nf4(W0_quantized, scales)

    # Step 2: Compute LoRA delta
    delta_W = (alpha / r) * (B @ A)

    # Step 3: Merge
    W_merged = W0_fp16 + delta_W

    return W_merged  # FP16 for saving
</syntaxhighlight>

'''Memory-Efficient Layer-by-Layer Processing:'''
<syntaxhighlight lang="python">
# Pseudo-code for memory-efficient export
def export_merged_model(model, output_dir, max_memory_usage=0.75):
    """
    Export merged model with controlled memory usage.
    Process one layer at a time to avoid OOM.
    """
    # Get available memory
    total_memory = get_gpu_memory()
    memory_limit = total_memory * max_memory_usage

    state_dict = {}
    for layer_name, (W0, A, B, alpha, r) in model.lora_layers():
        # Merge this layer
        W_merged = merge_lora(W0, A, B, alpha, r)

        # Move to CPU to free GPU memory
        state_dict[layer_name] = W_merged.cpu()

        # Clear GPU cache
        clear_gpu_cache()

    # Save with sharding
    save_sharded(state_dict, output_dir, max_shard_size="5GB")
</syntaxhighlight>

'''Safetensors Sharding:'''
Large models are split into multiple shard files with an index:

<syntaxhighlight lang="text">
model-00001-of-00003.safetensors  (5GB)
model-00002-of-00003.safetensors  (5GB)
model-00003-of-00003.safetensors  (2GB)
model.safetensors.index.json      (weight -> shard mapping)
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]

=== Tips and Tricks ===
