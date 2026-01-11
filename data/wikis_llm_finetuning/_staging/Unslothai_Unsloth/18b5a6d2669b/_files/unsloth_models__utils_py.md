# File: `unsloth/models/_utils.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 2453 |
| Classes | `HideLoggingMessage`, `_RaiseUninitialized`, `RaiseUninitialized`, `EmptyLogits`, `TorchAOConfig` |
| Functions | `extract_quant_model_param_count`, `get_model_param_count`, `patch_mistral_nemo_config`, `is_big_gpu`, `torch_compile_kwargs`, `patch_regional_compilation`, `prepare_model_for_kbit_training`, `has_internet`, `... +25 more` |
| Imports | accelerate, bitsandbytes, contextlib, dataclasses, device_type, functools, importlib, inspect, logging, math, ... +16 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core utilities module providing essential functions for model patching, preparation, quantization, gradient checkpointing, and training setup. Contains version detection, hardware capability checks, and low-level optimization utilities used across all model implementations.

**Mechanism:** Provides a comprehensive suite of utility functions including: hardware detection (bfloat16 support, GPU capabilities), model preparation (kbit training setup, gradient checkpointing patches), tokenizer fixes, statistics gathering, compilation patches for bitsandbytes and torch, RoPE scaling patches, offloading utilities, cross-entropy loss optimizations, and fast inference setup. Uses monkey patching extensively to override transformer library behaviors.

**Significance:** Foundation layer for the entire Unsloth optimization framework. These utilities enable core features like 4-bit training, custom gradient checkpointing, memory offloading, and model compilation. Nearly every model file imports from _utils, making it the most critical shared dependency in the codebase. Contains version "2026.1.2" and exports 70+ functions/classes.
