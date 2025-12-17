# File: `unsloth/models/loader_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 423 |
| Functions | `is_distributed`, `prepare_device_map`, `get_model_name` |
| Imports | device_type, gc, importlib, mapper, os, packaging, re, tempfile, torch, transformers, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Quantization and model name resolution utilities for flexible loading strategies.

**Mechanism:** Provides model name mapping (get_model_name), fp8 mode validation, offline FP8 quantization via torchao, dynamic model discovery, and cross-version compatibility checking with online model mapper updates.

**Significance:** Bridges between user model names and pre-quantized versions while supporting dynamic fp8 quantization for inference.
