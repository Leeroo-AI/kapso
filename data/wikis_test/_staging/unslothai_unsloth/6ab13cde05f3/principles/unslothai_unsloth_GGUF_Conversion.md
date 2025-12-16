{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|llama.cpp GGUF Spec|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
* [[source::Blog|GGUF Quantization Guide|https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html]]
* [[source::Doc|Ollama Modelfile|https://github.com/ollama/ollama/blob/main/docs/modelfile.md]]
|-
! Domains
| [[domain::Model_Export]], [[domain::Quantization]], [[domain::Deployment]], [[domain::Inference]]
|-
! Last Updated
| [[last_updated::2025-12-16 18:00 GMT]]
|}

== Overview ==

Model serialization format and conversion process for deploying Large Language Models with llama.cpp, Ollama, and compatible CPU/GPU inference engines.

=== Description ===

GGUF (GPT-Generated Unified Format) is a binary format designed for efficient LLM inference:

**Format Properties:**
- Single-file format containing weights, tokenizer, and metadata
- Architecture-agnostic structure supporting various model families
- Built-in quantization support for memory-efficient deployment
- Optimized for memory-mapped loading (mmap)

**Conversion Pipeline:**
1. Merge LoRA weights into base model (if applicable)
2. Convert to intermediate format (safetensors → GGUF)
3. Apply quantization (optional but recommended)
4. Package with tokenizer vocabulary and model metadata

**Quantization Options:**
Quantization reduces model size by representing weights with fewer bits:
- k-quants: Mixed-precision schemes that preserve important layers at higher precision
- Standard quants: Uniform bit-width across all layers

This enables running 7B models on laptops (8GB RAM) and 70B models on consumer GPUs (24GB VRAM).

=== Usage ===

Use GGUF conversion when:
* Deploying models for local inference with Ollama
* Running models on CPU or consumer hardware
* Distributing models in a portable single-file format
* Need fastest inference without Python runtime

Target environments:
- Ollama (easiest deployment)
- llama.cpp CLI or server
- llama-cpp-python bindings
- LM Studio, GPT4All, Jan.ai

== Theoretical Basis ==

=== GGUF File Structure ===

<syntaxhighlight lang="python">
# Abstract GGUF structure
class GGUF_File:
    magic: bytes = b"GGUF"
    version: int = 3

    # Metadata section
    metadata: Dict[str, Any] = {
        "general.architecture": "llama",
        "general.name": "model_name",
        "llama.attention.head_count": 32,
        "llama.context_length": 4096,
        # ... tokenizer info, quantization type, etc.
    }

    # Tensor data
    tensors: List[Tensor] = [
        # Each tensor has: name, shape, dtype, data
        {"name": "token_embd.weight", "shape": [32000, 4096], ...},
        {"name": "blk.0.attn_q.weight", "shape": [4096, 4096], ...},
        # ...
    ]
</syntaxhighlight>

=== k-Quant Mixed Precision ===

k-quants use different precision for different tensor types:

<syntaxhighlight lang="python">
# q4_k_m quantization scheme
QUANTIZATION_SCHEME = {
    "attention.wv": "Q6_K",      # Higher precision for value projection
    "feed_forward.w2": "Q6_K",   # Higher precision for down projection
    "default": "Q4_K",           # Standard 4-bit for other tensors
}

# This preserves quality in the most impactful layers
# while maximizing compression elsewhere
</syntaxhighlight>

=== Quantization Mathematics ===

For k-quant formats, weights are grouped into blocks:

<math>
Q_{block} = \text{round}\left(\frac{W_{block} - \min(W_{block})}{\max(W_{block}) - \min(W_{block})} \cdot (2^b - 1)\right)
</math>

Where:
- b: bit-width (2, 3, 4, 5, 6, or 8)
- Each block stores: quantized values + scale + min

'''Block quantization pseudo-code:'''
<syntaxhighlight lang="python">
def quantize_block(weights, block_size=32, bits=4):
    blocks = []
    for i in range(0, len(weights), block_size):
        block = weights[i:i+block_size]

        # Compute scale and zero-point
        w_min, w_max = block.min(), block.max()
        scale = (w_max - w_min) / (2**bits - 1)

        # Quantize
        quantized = ((block - w_min) / scale).round().astype(int)

        blocks.append({
            'data': quantized,
            'scale': scale,
            'min': w_min
        })

    return blocks
</syntaxhighlight>

=== Memory Requirements ===

Approximate VRAM/RAM needed for inference:

{| class="wikitable"
|-
! Model Size !! f16 !! q8_0 !! q5_k_m !! q4_k_m !! q2_k
|-
| 7B || 14 GB || 7.5 GB || 5.5 GB || 4.5 GB || 3 GB
|-
| 13B || 26 GB || 14 GB || 10 GB || 8 GB || 5.5 GB
|-
| 70B || 140 GB || 75 GB || 50 GB || 40 GB || 28 GB
|}

=== Conversion Pipeline ===

<syntaxhighlight lang="python">
# Abstract conversion flow
def convert_to_gguf(model_path, output_path, quant_method):
    # Step 1: Load model (HuggingFace format)
    weights = load_safetensors(model_path)
    tokenizer = load_tokenizer(model_path)
    config = load_config(model_path)

    # Step 2: Convert to intermediate f16 GGUF
    gguf_f16 = convert_hf_to_gguf(weights, tokenizer, config)

    # Step 3: Apply quantization
    if quant_method != "f16":
        gguf_quantized = quantize_gguf(gguf_f16, method=quant_method)
        return gguf_quantized

    return gguf_f16
</syntaxhighlight>

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:unslothai_unsloth_save_pretrained_gguf]]

=== Uses Heuristics ===
* [[uses_heuristic::Heuristic:unslothai_unsloth_Quantization_Method_Selection]]
