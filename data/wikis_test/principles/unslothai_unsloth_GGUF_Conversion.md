{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|GGML/GGUF Format Specification|https://github.com/ggml-org/ggml/blob/master/docs/gguf.md]]
* [[source::Repo|llama.cpp|https://github.com/ggml-org/llama.cpp]]
* [[source::Blog|Quantization Methods Comparison|https://ggml.ai/]]
|-
! Domains
| [[domain::Deep_Learning]], [[domain::Quantization]], [[domain::Model_Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-16 14:30 GMT]]
|}

== Overview ==
Format conversion and quantization process that transforms HuggingFace models into GGUF (GPT-Generated Unified Format) for efficient CPU/GPU inference with llama.cpp, Ollama, and other GGML-based runtimes.

=== Description ===
GGUF is a binary format designed for fast model loading and efficient inference, particularly on consumer hardware without dedicated ML accelerators. The conversion process involves:

1. **Format Translation**: Convert PyTorch tensors to GGML tensor format with embedded metadata
2. **Weight Quantization**: Apply various quantization schemes (Q4_K_M, Q8_0, etc.) to reduce model size
3. **Tokenizer Embedding**: Include vocabulary and tokenization rules in the model file
4. **Metadata Encoding**: Store model architecture, context length, and inference parameters

GGUF supports 20+ quantization methods with different quality/size tradeoffs:
- **Full precision**: F32, F16, BF16 - no quality loss
- **8-bit**: Q8_0, Q8_1 - minimal quality loss, ~2x compression
- **4-bit mixed**: Q4_K_M, Q5_K_M - good quality, ~4x compression
- **Aggressive**: Q3_K, Q2_K - noticeable quality loss, maximum compression

The "K" quantization methods use higher precision for "important" tensors (attention, first/last layers) while aggressively quantizing others.

=== Usage ===
Use GGUF conversion when:
- Deploying models for CPU inference (laptops, servers without GPUs)
- Running models on consumer GPUs with limited VRAM
- Distributing models for local deployment via Ollama
- Creating smaller model files for edge devices

Quantization method selection:
- **q4_k_m**: Best balance of size/quality (recommended default)
- **q5_k_m**: Higher quality, slightly larger
- **q8_0**: Near-lossless, 2x original size
- **f16**: No quality loss, largest files
- **q2_k/q3_k**: Aggressive compression, quality degradation

== Theoretical Basis ==
GGUF quantization uses block-wise quantization with learned or computed scales:

'''Block Quantization (Q4_0):'''
<math>
x_{quant} = \text{round}\left(\frac{x}{\text{scale}} \cdot 8\right), \quad \text{scale} = \frac{\max|x_{block}|}{7}
</math>

For a block of weights, compute a single scale factor, then quantize each weight to 4 bits.

'''K-Quant (Q4_K_M):'''
K-quantization uses super-blocks with multiple sub-blocks and importance weighting:

<syntaxhighlight lang="python">
# Pseudo-code for K-quantization
def quantize_k4_m(weights, block_size=32, super_block_size=256):
    """
    Q4_K_M: 4-bit with K-quant methodology.
    Uses 6-bit quants for important layers.
    """
    # Determine layer importance
    if is_important_layer(layer_name):
        return quantize_q6_k(weights)  # Higher precision

    # Standard 4-bit for other layers
    super_blocks = reshape(weights, [-1, super_block_size])
    quantized = []

    for super_block in super_blocks:
        # Compute super-block scale
        sb_scale = compute_scale(super_block)

        # Quantize sub-blocks within super-block
        sub_quants = []
        for sub_block in reshape(super_block, [-1, block_size]):
            sub_scale = compute_sub_scale(sub_block, sb_scale)
            quant = round(sub_block / sub_scale)
            sub_quants.append((quant, sub_scale))

        quantized.append((sub_quants, sb_scale))

    return quantized
</syntaxhighlight>

'''Important Layer Detection:'''
<syntaxhighlight lang="python">
# Pseudo-code for importance classification
def is_important_layer(name):
    """
    Layers that significantly impact model quality.
    Q4_K_M uses Q6_K for these.
    """
    important_patterns = [
        "attn.q",     # Query projection
        "attn.k",     # Key projection
        "output",     # Output embedding
        "token_embd", # Token embeddings
        "layers.0",   # First transformer block
        "layers.-1",  # Last transformer block
    ]
    return any(p in name for p in important_patterns)
</syntaxhighlight>

'''GGUF File Structure:'''
<syntaxhighlight lang="text">
GGUF File Layout:
┌─────────────────────────────┐
│ Magic: "GGUF" (4 bytes)     │
│ Version: 3 (4 bytes)        │
│ Tensor Count (8 bytes)      │
│ Metadata KV Count (8 bytes) │
├─────────────────────────────┤
│ Metadata Key-Value Pairs    │
│  - architecture: "llama"    │
│  - context_length: 4096     │
│  - tokenizer.ggml.model     │
│  - ...                      │
├─────────────────────────────┤
│ Tensor Info Array           │
│  - name, dims, type, offset │
├─────────────────────────────┤
│ Tensor Data (aligned)       │
│  - Quantized weight blocks  │
└─────────────────────────────┘
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Tips and Tricks ===
