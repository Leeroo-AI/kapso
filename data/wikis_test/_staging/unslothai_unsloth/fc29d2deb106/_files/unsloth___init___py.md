# File: `unsloth/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 280 |
| Imports | chat_templates, functools, import_fixes, importlib, inspect, models, numpy, os, packaging, re, ... +8 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization and environment setup for Unsloth library

**Mechanism:** This file serves as the main entry point for the Unsloth package. It performs critical initialization tasks: (1) Sets environment variables to indicate Unsloth's presence, (2) Checks for and warns about incorrect import order of critical modules (trl, transformers, peft) that need to be patched before import, (3) Verifies and validates required dependencies (unsloth_zoo, torch, bitsandbytes), (4) Detects device type (CUDA/HIP/XPU) and configures platform-specific settings including bfloat16 support, (5) Applies numerous compatibility patches for third-party libraries (xformers, vllm, datasets, etc.), (6) Handles CUDA library linking issues by running ldconfig when needed, and (7) Imports and re-exports all public APIs from submodules (models, save, chat_templates, tokenizer_utils, trainer).

**Significance:** This is the core initialization module that ensures Unsloth can properly optimize and patch the entire ML training stack. It's critical because Unsloth works by modifying transformers, peft, and trl at import time to provide 2x faster training. If these libraries are imported before Unsloth, the optimizations cannot be applied, leading to slower performance or OOM errors. The extensive environment detection and patching ensures compatibility across different hardware (NVIDIA/AMD/Intel GPUs), CUDA versions, and dependency versions.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
