# Principle: GGUF_Conversion

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|llama.cpp|https://github.com/ggerganov/llama.cpp]]
* [[source::Doc|GGUF Specification|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
* [[source::Repo|Unsloth|https://github.com/unslothai/unsloth]]
|-
! Domains
| [[domain::Model_Serialization]], [[domain::Quantization]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2026-01-12 00:00 GMT]]
|}

== Overview ==

Mechanism for converting HuggingFace models to GGUF format for deployment with llama.cpp, Ollama, and other inference engines.

=== Description ===

GGUF (GPT-Generated Unified Format) is a binary format designed for efficient inference of large language models. Converting to GGUF enables:

1. **CPU Inference**: Run models on CPU with reasonable performance
2. **Quantization**: Reduce model size by 2-8x with minimal quality loss
3. **Portability**: Single-file models that work across platforms
4. **Ollama Integration**: Deploy models locally via Ollama

The conversion process:
1. Merge LoRA adapters into base model (if needed)
2. Convert HuggingFace format to GGUF using llama.cpp's converter
3. Quantize to desired precision (q4_k_m, q8_0, etc.)
4. Generate Ollama Modelfile for easy deployment

=== Usage ===

Use this principle when:
* Deploying fine-tuned models for local inference
* Using llama.cpp, Ollama, or LM Studio
* Need to reduce model size for distribution
* Running inference on CPU or limited GPU memory

This step follows model training and merging.

== Theoretical Basis ==

'''GGUF File Structure:'''
<syntaxhighlight lang="text">
┌─────────────────┐
│    Header       │  Magic number, version, metadata count
├─────────────────┤
│   Metadata      │  Model config, tokenizer, architecture
├─────────────────┤
│  Tensor Info    │  Names, shapes, types, offsets
├─────────────────┤
│  Tensor Data    │  Quantized weight tensors
└─────────────────┘
</syntaxhighlight>

'''Conversion Pipeline:'''
<syntaxhighlight lang="python">
# Pseudo-code for GGUF conversion
def convert_to_gguf(model_dir, output_path, quantization):
    # Step 1: Convert HF to GGUF (f16/bf16)
    initial_gguf = llama_convert(model_dir, dtype="bf16")

    # Step 2: Quantize to target precision
    final_gguf = llama_quantize(initial_gguf, quantization)

    return final_gguf
</syntaxhighlight>

'''Quantization Trade-offs:'''
{| class="wikitable"
|-
! Method !! Bits/Weight !! Size Reduction !! Quality
|-
| f16/bf16 || 16 || 1x || 100%
|-
| q8_0 || 8 || 2x || ~99.5%
|-
| q6_k || 6.5 || 2.5x || ~99%
|-
| q5_k_m || 5.5 || 3x || ~98%
|-
| q4_k_m || 4.5 || 3.5x || ~97%
|-
| q4_0 || 4 || 4x || ~95%
|-
| q2_k || 2.5 || 6x || ~90%
|}

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_convert_to_gguf]]

