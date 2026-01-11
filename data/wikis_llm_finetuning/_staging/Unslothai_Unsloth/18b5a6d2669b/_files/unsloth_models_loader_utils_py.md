# File: `unsloth/models/loader_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 427 |
| Functions | `is_distributed`, `prepare_device_map`, `get_model_name` |
| Imports | device_type, gc, importlib, mapper, os, re, tempfile, torch, transformers, typing, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility functions for model name resolution, distributed training detection, device mapping, dtype selection, and FP8 quantization configuration. Supports the loader.py module with infrastructure for handling model variants and quantization schemes.

**Mechanism:** Key functions include: 1) get_model_name() which maps between full/4bit/fp8 model variants using mapper.py dictionaries (INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, FLOAT_TO_FP8_BLOCK_MAPPER), 2) is_distributed() which detects multi-GPU training by checking torch.distributed and environment variables (LOCAL_RANK, WORLD_SIZE), 3) prepare_device_map() for distributed device assignment, 4) _get_fp8_mode_and_check_settings() for FP8 quantization validation, 5) _offline_quantize_to_fp8() for model quantization. Includes BAD_MAPPINGS dict to handle problematic model names (e.g., redirecting oversized dynamic quants).

**Significance:** Critical glue layer between user-facing loader.py and low-level mapper.py. The model name resolution is essential for Unsloth's convenience feature of automatically selecting appropriate quantized variants. Distributed training detection enables multi-GPU workflows. FP8 support positions Unsloth for next-generation quantization. At 427 lines, it's moderately complex due to the variety of edge cases (distributed setups, version checks, platform differences).
