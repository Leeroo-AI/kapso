# Principle: LoRA Weight Merging

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Doc|PEFT Merging Documentation|https://huggingface.co/docs/peft/developer_guides/model_merging]]
* [[source::Blog|Understanding LoRA Merging|https://huggingface.co/blog/lora]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Deployment]], [[domain::Parameter_Efficient_Fine_Tuning]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Post-training process that combines Low-Rank Adaptation matrices with base model weights, eliminating adapter overhead and enabling efficient inference without architectural changes.

=== Description ===
LoRA weight merging is the process of integrating trained adapter weights back into the frozen base model. This is essential for:

1. **Deployment Simplification** - Single model file instead of base + adapter
2. **Inference Speed** - No extra computation for adapter forward pass
3. **Format Conversion** - Required before GGUF/ONNX export
4. **Quantization** - Merged weights can be quantized as a unit

'''Merging Process:'''
1. Load base model and LoRA adapter
2. For each target layer: W_merged = W_base + (B × A) × (α/r)
3. Replace layer weights with merged weights
4. Remove adapter components

'''Memory Considerations:'''
- Merging requires dequantization if base is in 4-bit
- Peak memory = full 16-bit model + original 4-bit model
- Unsloth handles this with progressive layer-by-layer merging

=== Usage ===
Merge LoRA weights when:
- Deploying to production inference
- Converting to GGUF format for llama.cpp/Ollama
- Sharing complete model on HuggingFace Hub
- Need maximum inference speed

'''Do NOT merge when:'''
- Multiple adapters need to be swapped at runtime
- Storage space is critical (adapter is much smaller)
- Continuing training from checkpoint

== Theoretical Basis ==
'''Merge Operation:'''

<math>
W_{merged} = W_0 + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
- W₀ ∈ ℝ^(d×k) is the frozen base weight
- A ∈ ℝ^(r×k) is the LoRA down-projection
- B ∈ ℝ^(d×r) is the LoRA up-projection
- α is the scaling factor, r is the rank

<syntaxhighlight lang="python">
def merge_lora_weights(base_weight, lora_A, lora_B, alpha, rank):
    """Merge LoRA adapter into base weights."""
    # Compute scaling factor
    scaling = alpha / rank

    # Compute low-rank update: B @ A gives (d, k) matrix
    delta_W = lora_B @ lora_A

    # Add scaled update to base
    merged = base_weight + scaling * delta_W

    return merged
</syntaxhighlight>

'''Memory-Efficient Merging (Unsloth approach):'''
<syntaxhighlight lang="python">
@torch.inference_mode()
def merge_lora_memory_efficient(model, temporary_dir, max_memory=0.9):
    """Merge LoRA weights with memory management."""

    # 1. Determine available memory
    free_gpu = get_free_gpu_memory()
    free_ram = get_free_ram()

    # 2. Process layer by layer to manage memory
    for layer_name, (W0, A, B) in get_lora_layers(model):

        # Dequantize base weight from 4-bit to 16-bit
        W0_fp16 = dequantize_nf4_to_fp16(W0)

        # Compute merged weight
        merged = W0_fp16 + (B @ A) * (alpha / r)

        # Handle memory overflow
        if not fits_in_memory(merged, free_gpu + free_ram):
            # Save to disk, load back when needed
            save_to_disk(merged, temporary_dir, layer_name)
        else:
            update_layer_weight(model, layer_name, merged)

        # Free intermediate tensors
        del W0_fp16
        torch.cuda.empty_cache()

    return model
</syntaxhighlight>

'''Dequantization for 4-bit Models:'''
<syntaxhighlight lang="python">
def dequantize_nf4_for_merge(quantized_weight, quant_state):
    """Dequantize NF4 weight to float16 for merging."""

    # NF4 stores:
    # - absmax: scaling factors per block
    # - quant_map: NF4 value lookup table
    # - code: quantized indices

    blocks = quantized_weight.reshape(-1, 64)  # 64 values per block
    output = torch.zeros_like(blocks, dtype=torch.float16)

    for i, block in enumerate(blocks):
        scale = quant_state.absmax[i]
        for j, code in enumerate(block):
            # Look up NF4 value and scale
            output[i, j] = NF4_VALUES[code] * scale

    return output.reshape(quantized_weight.shape)
</syntaxhighlight>

'''Save Methods:'''
| Method | Output | Use Case |
|--------|--------|----------|
| `lora` | adapter_model.safetensors | Continue training, small storage |
| `merged_16bit` | model.safetensors (fp16) | GGUF conversion, deployment |
| `merged_4bit` | model.safetensors (4-bit) | Continue 4-bit training |

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_unsloth_save_model]]
* [[implemented_by::Implementation:unslothai_unsloth_save_to_gguf]]

=== Tips and Tricks ===
