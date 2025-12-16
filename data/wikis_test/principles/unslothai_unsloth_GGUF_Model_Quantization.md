# Principle: GGUF Model Quantization

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GGUF Format Specification|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
* [[source::Repo|llama.cpp Quantization|https://github.com/ggerganov/llama.cpp]]
* [[source::Blog|GGUF Quantization Types|https://huggingface.co/docs/hub/gguf]]
|-
! Domains
| [[domain::Model_Compression]], [[domain::Quantization]], [[domain::Inference]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-15 20:00 GMT]]
|}

== Overview ==
Post-training quantization format that compresses model weights into various bit-width representations for efficient CPU and GPU inference, enabling local deployment of large language models on consumer hardware.

=== Description ===
GGUF (GPT-Generated Unified Format) is a binary format designed for efficient LLM inference with llama.cpp and compatible runtimes (Ollama, LM Studio, etc.). Key features:

1. **Mixed-Precision Quantization** - Different layers can use different quantization methods (e.g., Q6_K for critical layers, Q4_K for others)

2. **K-Quants Family** - Advanced quantization methods that preserve important weights with higher precision:
   - Q4_K_M: 4-bit with Q6_K for attention.wv and feed_forward.w2
   - Q5_K_M: 5-bit with Q6_K for same critical layers
   - Q6_K: 6-bit "super-blocks" for best 6-bit quality

3. **Single-File Format** - All weights, tokenizer, and metadata in one file for easy distribution

'''Trade-offs:'''
| Method | Bits | Quality | Size (7B) | Speed |
|--------|------|---------|-----------|-------|
| F16    | 16   | 100%    | 14GB      | Baseline |
| Q8_0   | 8    | ~99%    | 7GB       | Faster |
| Q5_K_M | 5-6  | ~97%    | 5GB       | Faster |
| Q4_K_M | 4-6  | ~95%    | 4GB       | Fastest |
| Q2_K   | 2-3  | ~85%    | 2.5GB     | Fastest |

=== Usage ===
Use GGUF quantization when:
- Deploying models for local inference (laptops, desktops)
- Running inference on CPU or consumer GPUs
- Model size/download time is a constraint
- Using Ollama, LM Studio, or llama.cpp for serving

'''Recommended Methods:'''
- **q4_k_m** - Best balance of quality and size (recommended default)
- **q5_k_m** - Higher quality with modest size increase
- **q8_0** - Near-lossless quality, reasonable compression
- **f16** - Maximum quality, for quality-critical applications

== Theoretical Basis ==
'''K-Quants Block Structure:'''

K-quantization methods use a hierarchical "super-block" structure for better precision:

<syntaxhighlight lang="python">
# Q4_K_M structure (simplified)
class Q4_K_Block:
    """Super-block containing 256 weights."""
    # Scale and min for the super-block (f16)
    d: float16      # Scale
    dmin: float16   # Min value

    # Per-block scales (6-bit each, packed)
    scales: uint8[12]  # Encodes 8 inner block scales

    # Quantized weights (4-bit each)
    qs: uint8[128]     # 256 weights, 2 per byte

def dequantize_q4_k(block):
    """Dequantize Q4_K block."""
    weights = []
    for i in range(8):  # 8 inner blocks of 32 weights
        scale = decode_scale(block.scales, i) * block.d
        min_val = decode_min(block.scales, i) * block.dmin

        for j in range(32):
            q = get_4bit_weight(block.qs, i * 32 + j)
            weights.append(scale * q + min_val)

    return weights
</syntaxhighlight>

'''Quantization Error Analysis:'''
<math>
Error = \sum_{i} (W_i - Q(W_i))^2
</math>

K-quants minimize this by:
1. Using higher precision for outlier-sensitive layers
2. Per-block scaling to adapt to local weight distributions
3. Asymmetric quantization (scale + offset) for better range utilization

'''Mixed Precision Selection:'''
<syntaxhighlight lang="python">
# Q4_K_M tensor assignment (llama.cpp logic)
def get_quant_type(tensor_name, base_method="Q4_K"):
    """Determine quantization type for each tensor."""

    # Critical tensors get higher precision
    if "attn_v.weight" in tensor_name:
        return "Q6_K"  # Value projections are sensitive
    elif "ffn_down.weight" in tensor_name:
        return "Q6_K"  # Feed-forward down projection
    elif "output.weight" in tensor_name:
        return "Q6_K"  # Final output layer
    else:
        return base_method  # Q4_K for most layers
</syntaxhighlight>

'''Conversion Pipeline:'''
1. Load HuggingFace model (16-bit)
2. Convert architecture to GGML format
3. Apply quantization method per-tensor
4. Write GGUF binary with metadata

== Related Pages ==
=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_to_gguf]]
* [[implemented_by::Implementation:unslothai_unsloth_OLLAMA_TEMPLATES]]

=== Tips and Tricks ===
