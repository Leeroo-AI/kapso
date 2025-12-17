# File: `unsloth/models/mapper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1324 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive bidirectional mapping dictionary between 4-bit quantized models and float versions with fp8 variants.

**Mechanism:** Maintains __INT_TO_FLOAT_MAPPER (1000+ entries) mapping 4bit model names to full-precision alternatives, generates derived mappings for 8bit/16bit versions, and includes model-specific fp8 block/row granularity mappings for dynamic quantization.

**Significance:** Enables automatic model variant selection and supports on-the-fly quantization strategies for inference optimization. Core lookup table for model resolution.
