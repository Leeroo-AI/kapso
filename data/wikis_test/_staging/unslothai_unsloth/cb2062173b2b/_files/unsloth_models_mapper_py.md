# File: `unsloth/models/mapper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1324 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model name mapping database that provides bidirectional mappings between quantized and full-precision model variants, enabling automatic model variant resolution during loading.

**Mechanism:** Defines a large dictionary __INT_TO_FLOAT_MAPPER containing mappings from quantized model names (e.g., "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit") to their full-precision equivalents (e.g., "meta-llama/Llama-3.2-1B-Instruct"). Supports multiple mapping types: simple 4-bit to float16 mappings, dict-based mappings with separate "8" (FP8) and "16" (float16) entries for models with multiple quantization options. Processes mappings into five dictionaries: INT_TO_FLOAT_MAPPER (4-bit to float), FLOAT_TO_INT_MAPPER (float to 4-bit), MAP_TO_UNSLOTH_16bit (official to Unsloth 16-bit), FLOAT_TO_FP8_BLOCK_MAPPER, and FLOAT_TO_FP8_ROW_MAPPER for FP8 variants. Covers hundreds of models including Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, Granite, and vision models.

**Significance:** Essential database that enables Unsloth's smart model loading. Users can specify either quantized or full-precision names and the loader automatically resolves to the appropriate variant based on quantization settings. This provides a seamless user experience and enables pre-quantized model downloads for faster loading. The comprehensive mapping coverage supports Unsloth's wide model compatibility while maintaining a simple API.
