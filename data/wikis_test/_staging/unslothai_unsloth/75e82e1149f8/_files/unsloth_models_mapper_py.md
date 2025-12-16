# File: `unsloth/models/mapper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1324 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Large dictionary mappings between model name variants

**Mechanism:** Defines several large Python dictionaries that map between different versions of the same model:
- **INT_TO_FLOAT_MAPPER**: Maps 4bit quantized model names to their float16/bfloat16 equivalents
- **FLOAT_TO_INT_MAPPER**: Reverse mapping from float models to 4bit versions
- **MAP_TO_UNSLOTH_16bit**: Maps to Unsloth's optimized 16bit versions
- **FLOAT_TO_FP8_BLOCK_MAPPER**: Maps to FP8 block-wise quantized versions
- **FLOAT_TO_FP8_ROW_MAPPER**: Maps to FP8 row-wise quantized versions

Contains hundreds of entries for popular models like Llama, Mistral, Gemma, Qwen, Phi, etc., mapping between HuggingFace official names and Unsloth's pre-quantized variants.

**Significance:** Data file that enables automatic model variant resolution. When users request a model, loader_utils.py uses these dictionaries to find the optimal version. This allows Unsloth to:
- Automatically load faster pre-quantized models
- Convert between quantization formats seamlessly
- Support both HuggingFace and Unsloth model naming conventions
- Provide backwards compatibility as new models are added

The mappings are periodically updated and can be fetched from GitHub for the latest supported models.
