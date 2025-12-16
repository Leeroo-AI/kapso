# File: `unsloth/models/loader_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 423 |
| Functions | `is_distributed`, `prepare_device_map`, `get_model_name` |
| Imports | device_type, gc, importlib, mapper, os, packaging, re, tempfile, torch, transformers, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model name resolution and quantization configuration utilities

**Mechanism:** Provides critical functions for model loading:
- **Model name mapping**: Resolves model names to their optimized variants (e.g., mapping float16 models to pre-quantized 4bit versions, or vice versa)
- **Distributed training setup**: Detects multi-GPU/multi-node environments and prepares appropriate device maps
- **FP8 quantization**: Handles FP8 model loading with block-wise or row-wise quantization modes, including offline quantization when pre-quantized versions aren't available
- **Version compatibility**: Checks if transformers version supports certain features (4bit loading, etc.)

Uses the `mapper.py` dictionaries (INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, etc.) to translate between model name variants. Can fetch updated mappings from GitHub if needed.

**Significance:** Essential for Unsloth's model variant management system. Enables users to request "meta-llama/Llama-3-8B" and automatically load "unsloth/Llama-3-8B-bnb-4bit" when `load_in_4bit=True`, or vice versa. Handles edge cases like:
- Bad model name mappings (too-large dynamic quants)
- Distributed training device placement
- FP8 quantization modes (FBGEMM vs TorchAO)
- Pre-quantized model availability checks
