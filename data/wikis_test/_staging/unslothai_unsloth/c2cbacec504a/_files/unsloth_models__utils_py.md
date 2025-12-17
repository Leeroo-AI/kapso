# File: `unsloth/models/_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2346 |
| Classes | `HideLoggingMessage`, `_RaiseUninitialized`, `RaiseUninitialized`, `EmptyLogits`, `TorchAOConfig` |
| Functions | `extract_quant_model_param_count`, `get_model_param_count`, `patch_mistral_nemo_config`, `is_big_gpu`, `torch_compile_kwargs`, `patch_regional_compilation`, `prepare_model_for_kbit_training`, `has_internet`, `... +23 more` |
| Imports | accelerate, bitsandbytes, contextlib, dataclasses, device_type, functools, importlib, inspect, logging, math, ... +16 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core utility functions for model patching, compilation, quantization validation, and training enhancements.

**Mechanism:** Contains 50+ functions including: parameter counting (extract_quant_model_param_count), device management (move_to_device, offload_to_disk), RoPE scaling patches (patch_linear_scaling), gradient accumulation fixes, compiler optimization helpers, QAT preparation, FP8 support verification, and TorchAO configuration handling.

**Significance:** Infrastructure layer enabling efficient training through memory optimization, compilation patching, and quantization management. Used throughout the codebase for common operations.
