# File: `unsloth/models/mapper.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1329 |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines comprehensive mapping dictionaries that translate between full precision model names and their quantized variants (4-bit, FP8 block-wise, FP8 row-wise) across the entire Unsloth model catalog. Enables automatic model variant selection.

**Mechanism:** Exports five key dictionaries: 1) INT_TO_FLOAT_MAPPER: maps "unsloth/model-bnb-4bit" -> tuple of equivalent full models, 2) FLOAT_TO_INT_MAPPER: reverse mapping from full to 4-bit variants, 3) MAP_TO_UNSLOTH_16bit: maps to unsloth-optimized 16-bit models, 4) FLOAT_TO_FP8_BLOCK_MAPPER: maps to block-wise FP8 quantized models, 5) FLOAT_TO_FP8_ROW_MAPPER: maps to row-wise FP8 models. Contains hundreds of entries covering LLaMA, Mistral, Qwen, Gemma, Phi, and many other model families at various sizes (1B to 405B parameters).

**Significance:** The "phone book" of Unsloth's model ecosystem. At 1329 lines, this is mostly data (dictionary definitions), not code. Enables the key user experience where specifying "meta-llama/Llama-2-7b-hf" with load_in_4bit=True automatically loads "unsloth/llama-2-7b-bnb-4bit". Maintenance burden: must be updated for every new model release. The FP8 mappers show Unsloth's forward-looking quantization strategy.
