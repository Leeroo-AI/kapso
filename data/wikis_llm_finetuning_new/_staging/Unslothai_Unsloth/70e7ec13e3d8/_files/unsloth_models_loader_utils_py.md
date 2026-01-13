# File: `unsloth/models/loader_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 427 |
| Functions | `is_distributed`, `prepare_device_map`, `get_model_name` |
| Imports | device_type, gc, importlib, mapper, os, re, tempfile, torch, transformers, typing, ... +2 more |

## Understanding

**Status:** Explored

**Purpose:** Utility functions for model loading including model name resolution, FP8 quantization support, and distributed training device mapping.

**Mechanism:** `get_model_name()` resolves user-provided model names to optimized Unsloth variants using mappings from `mapper.py`, handling 4-bit to 16-bit conversions and FP8 model lookups. `_offline_quantize_to_fp8()` performs on-the-fly FP8 quantization using TorchAO when pre-quantized models aren't available, saving to a temp directory. `_get_fp8_mode_and_check_settings()` validates FP8 requirements (H100+ GPU, torch 2.9+, torchao 0.15+). `prepare_device_map()` and `is_distributed()` handle multi-GPU training setup. Includes `BAD_MAPPINGS` dict to override problematic model mappings (e.g., MoE models that load slowly).

**Significance:** Supporting utility - centralizes model name resolution and quantization logic used by the main loader. Essential for the seamless model mapping that allows users to specify any model name and automatically get optimized variants.
