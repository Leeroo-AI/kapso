{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Paper|QLoRA: Efficient Finetuning of Quantized LLMs|https://arxiv.org/abs/2305.14314]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::PEFT]], [[domain::Model_Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Mathematical operation that fuses trained low-rank adapter matrices into base model weights, creating a standalone model that performs identically to the adapter-augmented version without runtime overhead.

=== Description ===
LoRA merging is the process of combining the learned adapter weights (A, B matrices) with the frozen base model weights. This is essential for deployment because:

1. **No PEFT Dependency**: Merged models can be loaded with standard transformers without PEFT library
2. **Zero Inference Overhead**: No additional matrix multiplications during forward pass
3. **Format Compatibility**: Merged weights can be converted to other formats (GGUF, ONNX)
4. **Simplified Deployment**: Single model file instead of base + adapter

The merge is mathematically exact - the merged model produces identical outputs to the base + adapter model. For quantized models (4-bit QLoRA), merging involves dequantization since the adapter's contribution cannot be stored in the quantized format.

The key challenge in merging is memory management: dequantizing a 7B model from 4-bit to FP16 temporarily requires ~14GB of RAM/VRAM. Layer-by-layer processing and memory-efficient algorithms address this constraint.

=== Usage ===
Perform LoRA merging when:
- Preparing a model for production deployment
- Converting to deployment formats (GGUF, TensorRT)
- Distributing a model without requiring PEFT
- Creating a base model for further fine-tuning (DPO, continued SFT)

Keep adapters separate when:
- Running multiple task-specific adapters on shared base model
- Storage space is critical (adapters are ~100MB vs full model)
- You need to quickly switch between fine-tuned variants
- Continuing iterative fine-tuning experiments

== Theoretical Basis ==
For a layer with base weights W₀ and LoRA adapters A, B, the merge operation computes:

<math>
W_{merged} = W_0 + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
- <math>W_0 \in \mathbb{R}^{d \times k}</math>: Frozen base weights
- <math>A \in \mathbb{R}^{r \times k}</math>: Up-projection (initialized random)
- <math>B \in \mathbb{R}^{d \times r}</math>: Down-projection (initialized zeros)
- <math>\alpha</math>: LoRA alpha (scaling hyperparameter)
- <math>r</math>: LoRA rank

'''Merge for FP16 Base Model:'''
<syntaxhighlight lang="python">
# Pseudo-code for standard LoRA merge
def merge_lora_fp16(W0, A, B, alpha, r):
    """
    Merge LoRA into FP16 base weights.
    Straightforward matrix computation.
    """
    # Compute scaled LoRA delta
    scale = alpha / r
    delta_W = scale * (B @ A)

    # Add to base weights
    W_merged = W0 + delta_W

    return W_merged
</syntaxhighlight>

'''Merge for 4-bit Quantized Base Model:'''
<syntaxhighlight lang="python">
# Pseudo-code for 4-bit LoRA merge (dequantize-merge)
def merge_lora_4bit(W0_quantized, quantization_state, A, B, alpha, r):
    """
    Merge LoRA into 4-bit NF4 quantized base weights.
    Must dequantize first since 4-bit cannot represent merged values.
    """
    # Step 1: Dequantize base weights to FP16
    # NF4 uses block-wise scales
    W0_fp16 = dequantize_nf4(
        W0_quantized,
        scales=quantization_state.scales,
        absmax=quantization_state.absmax,
        blocksize=quantization_state.blocksize
    )

    # Step 2: Compute LoRA contribution (in FP16)
    scale = alpha / r
    delta_W = scale * (B.float() @ A.float())

    # Step 3: Merge
    W_merged = W0_fp16 + delta_W.to(W0_fp16.dtype)

    return W_merged  # FP16, ready for saving or re-quantization
</syntaxhighlight>

'''Memory-Efficient Layer-by-Layer Merge:'''
<syntaxhighlight lang="python">
# Pseudo-code for memory-efficient merging
def merge_model_efficiently(model, max_memory_fraction=0.75):
    """
    Merge LoRA adapters with controlled memory usage.
    Process one layer at a time, immediately moving to CPU.
    """
    merged_state_dict = {}

    for layer_name, layer in model.named_modules():
        if not has_lora(layer):
            # Non-LoRA layer: just copy weights
            merged_state_dict[layer_name] = layer.weight.cpu()
            continue

        # Get LoRA components
        W0 = layer.base_layer.weight
        A = layer.lora_A.weight
        B = layer.lora_B.weight
        scale = layer.scaling

        # Check if base is quantized
        if is_quantized(W0):
            W0_fp16 = dequantize(W0)
        else:
            W0_fp16 = W0.float()

        # Merge
        W_merged = W0_fp16 + scale * (B.float() @ A.float())

        # Immediately move to CPU to free GPU memory
        merged_state_dict[layer_name] = W_merged.cpu()

        # Clear CUDA cache
        torch.cuda.empty_cache()

    return merged_state_dict
</syntaxhighlight>

'''Mathematical Equivalence:'''
The merge is exact because LoRA modifies the forward pass additively:

<syntaxhighlight lang="text">
Original forward (with LoRA):
    h = W0 @ x + (scale * B @ A) @ x
    h = W0 @ x + scale * B @ (A @ x)

Merged forward:
    h = W_merged @ x
    h = (W0 + scale * B @ A) @ x
    h = W0 @ x + scale * (B @ A) @ x

Both produce identical output h for any input x.
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Tips and Tricks ===
