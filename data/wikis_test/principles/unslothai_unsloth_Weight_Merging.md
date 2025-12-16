{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Paper|LoRA: Low-Rank Adaptation of Large Language Models|https://arxiv.org/abs/2106.09685]]
* [[source::Blog|Unsloth Save Guide|https://docs.unsloth.ai/basics/running-and-saving-models]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Model_Export]], [[domain::LoRA]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Technique for fusing Low-Rank Adaptation (LoRA) adapter weights with base model weights to produce a standalone model without adapter dependencies.

=== Description ===

Weight merging is the process of combining LoRA adapter matrices back into the original model weights, producing a model that:

* **Eliminates adapter overhead**: No runtime LoRA computation during inference
* **Removes PEFT dependency**: Merged model loads with standard HuggingFace transformers
* **Enables format conversion**: Required step before GGUF, ONNX, or TensorRT conversion

The challenge is that base weights are typically stored in 4-bit quantized format, requiring:
1. Dequantization to full precision
2. LoRA weight fusion
3. Re-serialization (optionally re-quantized)

Unsloth implements memory-efficient chunked merging that processes one layer at a time, enabling 70B+ model merging on consumer GPUs.

=== Usage ===

Apply weight merging when:
* Preparing models for deployment (vLLM, SGLang, TGI)
* Converting to GGUF format for llama.cpp or Ollama
* Creating standalone HuggingFace models for distribution
* Eliminating PEFT library dependency from inference pipeline

Note: Merging is a one-way operation. Keep your LoRA adapters saved separately if you may need to continue training.

== Theoretical Basis ==

=== LoRA Weight Fusion ===

The merge operation combines base weights with adapter matrices:

<math>
W_{merged} = W_0 + \frac{\alpha}{r} \cdot B \cdot A
</math>

Where:
- W₀: Original model weights (dequantized from 4-bit if necessary)
- B ∈ ℝᵈˣʳ: LoRA down-projection matrix
- A ∈ ℝʳˣᵏ: LoRA up-projection matrix
- α: LoRA scaling factor
- r: LoRA rank

=== Dequantization Process ===

For 4-bit NF4 quantized models:

<syntaxhighlight lang="python">
# Abstract dequantization process
def dequantize_nf4(indices, scale, nf4_quantiles):
    # indices: 4-bit quantized values
    # scale: per-block scaling factor

    # Look up quantile values
    values = nf4_quantiles[indices]

    # Apply scale to recover original range
    dequantized = values * scale

    return dequantized.to(torch.float32)
</syntaxhighlight>

=== Memory-Efficient Merging ===

To avoid loading entire model into GPU memory:

<syntaxhighlight lang="python">
# Abstract layer-by-layer merge process
def merge_model_efficiently(model, max_memory=0.9):
    for layer_idx, layer in enumerate(model.layers):
        for name, module in layer.named_modules():
            if has_lora(module):
                # 1. Get base weights
                W_base = dequantize(module.weight)

                # 2. Get LoRA weights
                A, B, scale = get_lora_params(module)

                # 3. Compute merged weights
                W_merged = W_base + scale * (B @ A)

                # 4. Store merged weights
                save_weight(W_merged, layer_idx, name)

                # 5. Free memory
                del W_base, W_merged
                torch.cuda.empty_cache()

        # Check memory threshold
        if gpu_memory_usage() > max_memory:
            flush_to_disk()
</syntaxhighlight>

=== Precision Considerations ===

The merge process involves multiple precision conversions:

{| class="wikitable"
|-
! Stage !! Precision !! Reason
|-
| Base weights (stored) || 4-bit NF4 || Memory efficiency during training
|-
| Dequantized || float32 || Full precision for accurate merge
|-
| LoRA weights || float16/bfloat16 || Training precision
|-
| Merge operation || float32 || Avoid precision loss in addition
|-
| Output (16-bit) || float16/bfloat16 || Standard deployment format
|-
| Output (4-bit) || 4-bit NF4 || If re-quantizing for inference
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_merged]]
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Implementation Mapping ===

This principle is implemented across multiple APIs:

{| class="wikitable"
|-
! Concept Component !! Implementation !! What It Does
|-
| LoRA weight fusion (W = W₀ + αBA) || [[Implementation:unslothai_unsloth_save_pretrained_merged]] || Merges adapters into base weights, saves as HuggingFace safetensors
|-
| LoRA merge + GGUF quantization || [[Implementation:unslothai_unsloth_save_pretrained_gguf]] || Merges adapters, then converts to GGUF format with quantization
|}

'''When to use each:'''
* Use `save_pretrained_merged` when deploying to vLLM, SGLang, or HuggingFace Hub
* Use `save_pretrained_gguf` when deploying to Ollama or llama.cpp
