# File: `unsloth/models/mapper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1329 |

## Understanding

**Status:** Explored

**Purpose:** Defines comprehensive model name mapping dictionaries that translate between various model identifiers (4-bit, 16-bit, FP8, official, Unsloth-optimized versions).

**Mechanism:** Contains a large `__INT_TO_FLOAT_MAPPER` dictionary mapping Unsloth 4-bit model names to their 16-bit equivalents and official HuggingFace names. Some entries use nested dicts for FP8 variants (keys "8" and "16"). Post-processing loop generates derived mappings: `INT_TO_FLOAT_MAPPER` (4-bit to 16-bit), `FLOAT_TO_INT_MAPPER` (reverse), `MAP_TO_UNSLOTH_16bit` (official to Unsloth 16-bit), `FLOAT_TO_FP8_BLOCK_MAPPER` and `FLOAT_TO_FP8_ROW_MAPPER` (for FP8 quantization modes). Covers extensive model families: Llama (2, 3, 3.1, 3.2, 3.3), Mistral, Qwen (2, 2.5, 3), Gemma (1, 2, 3, 3N), Phi (3, 4), DeepSeek-R1, Granite, and many more.

**Significance:** Data configuration - this is a pure data file that enables Unsloth's model name aliasing system. Critical for user convenience, allowing `unsloth/Qwen3-8B-bnb-4bit` to resolve correctly to pre-quantized or official models based on loading options.
