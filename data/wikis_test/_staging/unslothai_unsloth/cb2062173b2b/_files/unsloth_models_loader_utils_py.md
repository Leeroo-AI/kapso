# File: `unsloth/models/loader_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 423 |
| Functions | `is_distributed`, `prepare_device_map`, `get_model_name` |
| Imports | device_type, gc, importlib, mapper, os, packaging, re, tempfile, torch, transformers, ... +3 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for model loading operations including distributed training detection, device mapping, model name resolution for pre-quantized variants, and FP8 quantization support.

**Mechanism:** Provides three key functions: (1) is_distributed - detects distributed training by checking torch.distributed state and environment variables (LOCAL_RANK, WORLD_SIZE), (2) prepare_device_map - creates device mapping for distributed setups and sets appropriate device, (3) get_model_name - resolves model names to appropriate variants (4-bit, FP8, 16-bit) using mapper dictionaries, handles bad mappings that need redirection. Additionally includes FP8 quantization utilities: _get_torchao_fp8_config creates Float8 configs with row/block granularity, _offline_quantize_to_fp8 quantizes models offline using TorchAO and saves to temp directory, _tag_model_with_fp8_torchao_config tags models with FP8 config, _get_fp8_mode_and_check_settings validates FP8 settings and environment requirements (H100+ GPU, torch 2.9+, torchao 0.15+).

**Significance:** Critical utility module that handles model loading logistics. The model name resolution enables automatic selection of optimized pre-quantized models, reducing download sizes and loading times. The distributed training utilities ensure proper device placement in multi-GPU setups. FP8 quantization support enables memory-efficient inference on modern GPUs while validating hardware and software requirements.
