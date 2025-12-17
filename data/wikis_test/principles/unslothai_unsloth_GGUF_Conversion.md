# Principle: unslothai_unsloth_GGUF_Conversion

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|GGUF Specification|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
* [[source::Blog|GGML Quantization|https://huggingface.co/blog/overview-quantization-llm]]
|-
! Domains
| [[domain::NLP]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 15:00 GMT]]
|}

== Overview ==

Technique for converting language models to the GGUF format optimized for CPU inference and edge deployment.

=== Description ===

GGUF (GPT-Generated Unified Format) is the standard format for llama.cpp and compatible inference engines. It provides:

1. **Single-file distribution**: Model + tokenizer + metadata in one file
2. **Flexible quantization**: Multiple precision levels from f16 to 2-bit
3. **CPU optimization**: SIMD-accelerated inference without GPU
4. **Broad compatibility**: Works with Ollama, llama.cpp, LM Studio, etc.

GGUF conversion is essential for deploying models outside the Python/CUDA ecosystem.

=== Usage ===

Use GGUF conversion when:
- Deploying to servers without GPUs
- Running models on laptops/edge devices
- Distributing models in a portable format
- Using Ollama or similar tools

== Theoretical Basis ==

=== GGUF File Structure ===

<syntaxhighlight lang="python">
# GGUF file layout
gguf_structure = {
    "header": {
        "magic": "GGUF",
        "version": 3,
        "tensor_count": N,
        "metadata_count": M,
    },
    "metadata": {
        "general.architecture": "llama",
        "llama.context_length": 4096,
        "tokenizer.ggml.tokens": [...],
        # ... more metadata
    },
    "tensors": [
        {"name": "blk.0.attn_q.weight", "shape": [...], "type": "Q4_K"},
        # ... all model weights
    ],
}
</syntaxhighlight>

=== Quantization Types ===

GGUF supports multiple quantization schemes:

{| class="wikitable"
|-
! Type !! Bits/Weight !! Method !! Use Case
|-
| F16 || 16 || Float16 || Maximum quality
|-
| Q8_0 || 8 || Round to nearest || Near-lossless
|-
| Q6_K || 6.5 || K-quant super-block || High quality
|-
| Q5_K || 5.5 || K-quant super-block || Balanced
|-
| Q4_K || 4.5 || K-quant super-block || Good default
|-
| Q4_0 || 4 || Uniform || Fast, lower quality
|-
| Q3_K || 3.5 || K-quant super-block || Aggressive compression
|-
| Q2_K || 2.5 || K-quant super-block || Maximum compression
|}

=== K-Quantization ===

K-quants use non-uniform quantization:

'''Pseudo-code Logic:'''
<syntaxhighlight lang="python">
def k_quant_block(weights, block_size=32):
    """
    K-quantization uses super-blocks with mixed precision.
    """
    # Split into blocks
    blocks = weights.reshape(-1, block_size)

    quantized_blocks = []
    for block in blocks:
        # Compute optimal scale and zero point
        scale, zero = compute_optimal_params(block)

        # Some weights get higher precision (importance-based)
        important_weights = identify_important(block)

        # Quantize with mixed precision
        quant_block = {
            "scale": scale,
            "zero": zero,
            "values": quantize_mixed(block, important_weights),
        }
        quantized_blocks.append(quant_block)

    return quantized_blocks
</syntaxhighlight>

=== Quality vs Size Tradeoff ===

Perplexity increase by quantization method (approximate):

<syntaxhighlight lang="python">
# Perplexity increase vs f16 baseline
perplexity_delta = {
    "f16":    0.00,  # Baseline
    "q8_0":   0.01,  # Negligible
    "q6_k":   0.05,  # Very small
    "q5_k_m": 0.10,  # Small
    "q4_k_m": 0.20,  # Acceptable
    "q4_0":   0.30,  # Noticeable
    "q3_k_m": 0.50,  # Significant
    "q2_k":   1.00,  # Large
}
</syntaxhighlight>

=== Conversion Pipeline ===

<syntaxhighlight lang="python">
def gguf_conversion_pipeline(hf_model_path, output_path, quant_method):
    """
    Full GGUF conversion process.
    """
    # 1. Convert HF format to GGUF
    # Uses llama.cpp's convert_hf_to_gguf.py
    initial_gguf = convert_hf_to_gguf(
        hf_model_path,
        output_type="f16",  # Start with f16
    )

    # 2. Quantize if not f16
    if quant_method != "f16":
        final_gguf = quantize_gguf(
            initial_gguf,
            output_path,
            quant_type=quant_method,
        )
    else:
        final_gguf = initial_gguf

    # 3. Validate
    validate_gguf(final_gguf)

    return final_gguf
</syntaxhighlight>

== Practical Guide ==

=== Choosing Quantization ===

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Quality-first | q8_0 | Near-lossless |
| Balanced | q4_k_m | Best quality/size ratio |
| Size-constrained | q4_k_s or q3_k_m | Smaller files |
| Mobile/edge | q4_0 | Fastest inference |

=== Memory Requirements ===

<syntaxhighlight lang="python">
# RAM needed to run GGUF models
def estimate_ram(model_params_billions, quant_bits):
    # Base model size
    model_size_gb = model_params_billions * quant_bits / 8

    # Context buffer (~2GB for 4K context)
    context_buffer = 2.0

    # Overhead
    overhead = 0.5

    return model_size_gb + context_buffer + overhead

# Examples:
# 7B q4_k_m: 7 * 0.5 + 2 + 0.5 = ~6 GB RAM
# 7B q8_0: 7 * 1 + 2 + 0.5 = ~9.5 GB RAM
# 13B q4_k_m: 13 * 0.5 + 2 + 0.5 = ~9 GB RAM
</syntaxhighlight>

=== Validation After Conversion ===

<syntaxhighlight lang="bash">
# Validate GGUF file structure
./llama-cli -m model.gguf --info

# Test generation
./llama-cli -m model.gguf -p "Test prompt" -n 20

# Check perplexity (if you have test data)
./llama-perplexity -m model.gguf -f test.txt
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Used In Workflows ===
* [[used_by::Workflow:unslothai_unsloth_Model_Export]]
