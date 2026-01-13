# File: `unsloth/registry/_llama.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 125 |
| Classes | `LlamaModelInfo`, `LlamaVisionModelInfo` |
| Functions | `register_llama_3_1_models`, `register_llama_3_2_models`, `register_llama_3_2_vision_models`, `register_llama_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines model metadata and registration logic for Meta's Llama model family including Llama 3.1, 3.2, and 3.2 Vision variants.

**Mechanism:** Provides two `ModelInfo` subclasses: `LlamaModelInfo` for text models (`Llama-{version}-{size}B`) and `LlamaVisionModelInfo` for vision models (`Llama-{version}-{size}B-Vision`). Defines `ModelMeta` configurations for Llama 3.1 (8B), Llama 3.2 base/instruct (1B, 3B), and Llama 3.2 Vision (11B, 90B). Vision models are marked multimodal. Uses size-specific quantization: 11B vision supports full quantization options while 90B only supports unquantized.

**Significance:** Enables Unsloth support for the Llama model family with correct path construction. Demonstrates the registry's ability to handle different model architectures (text vs vision) within the same family and size-dependent quantization availability.
