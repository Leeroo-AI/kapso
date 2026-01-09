# Principle: GGUF_Export

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|GGUF Specification|https://github.com/ggerganov/ggml/blob/master/docs/gguf.md]]
* [[source::Doc|llama.cpp|https://github.com/ggerganov/llama.cpp]]
|-
! Domains
| [[domain::Model_Export]], [[domain::GGUF]], [[domain::Quantization]]
|-
! Last Updated
| [[last_updated::2026-01-09 16:00 GMT]]
|}

== Overview ==

Converting fine-tuned models to GGUF format for deployment with llama.cpp, Ollama, and other GGML-based inference engines.

=== Description ===

GGUF Export converts HuggingFace models to GGUF (GPT-Generated Unified Format):

1. **Model Serialization**: Convert weights to GGUF tensor format
2. **Tokenizer Embedding**: Embed tokenizer vocabulary and special tokens
3. **Quantization**: Apply selected quantization method
4. **Metadata**: Include model architecture and hyperparameters

=== Usage ===

Use GGUF Export when deploying fine-tuned models for:
* Local inference with llama.cpp
* Ollama deployment
* Edge device deployment
* CPU-only inference

== Theoretical Basis ==

=== GGUF Structure ===

GGUF files contain:
* Magic number and version
* Tensor count and metadata count
* Key-value metadata (architecture, hyperparameters)
* Tensor info (names, shapes, dtypes, offsets)
* Tensor data (aligned to 32 bytes)

=== Conversion Pipeline ===

1. **HF → Float16**: Dequantize 4-bit, merge LoRA
2. **Float16 → GGUF**: Convert using llama.cpp's convert.py
3. **GGUF → Quantized GGUF**: Apply llama-quantize

=== Architecture Support ===

Supported architectures:
* LLaMA, Mistral, Mixtral
* Qwen, Qwen2
* Gemma, Gemma2
* Phi, Phi-3
* LLaVA (vision models - partial)

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Unslothai_Unsloth_save_to_gguf]]

