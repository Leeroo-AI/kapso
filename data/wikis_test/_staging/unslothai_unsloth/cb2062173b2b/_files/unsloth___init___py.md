# File: `unsloth/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 280 |
| Imports | chat_templates, functools, import_fixes, importlib, inspect, models, numpy, os, packaging, re, ... +8 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization and bootstrap module that orchestrates the setup of Unsloth's optimized training environment. Sets critical environment variables, validates dependencies, applies compatibility patches, and exports the public API for model loading, training, and utilities.

**Mechanism:**
- Sets `UNSLOTH_IS_PRESENT` environment variable to signal presence
- Validates import order by checking if `trl`, `transformers`, or `peft` were imported before Unsloth (warns if optimizations won't apply)
- Applies import fixes via `import_fixes` module (protobuf, fbgemm_gpu, xformers, vLLM, etc.)
- Detects device type (CUDA/HIP/XPU) and configures bfloat16 support detection
- Handles CUDA/bitsandbytes linking issues through `ldconfig` system calls
- Checks and validates `unsloth_zoo` dependency version (minimum 2025.12.4)
- Patches PyTorch's `is_bf16_supported` function for compatibility across versions
- Imports and re-exports public API from submodules (models, save, chat_templates, tokenizer_utils, trainer, RL environments)
- Calls `_patch_trl_trainer()` to apply backward compatibility patches for TRL

**Significance:** This is the entry point for the entire Unsloth library. It's critical because it ensures the correct initialization order, applies necessary patches before other libraries are loaded, and validates the environment. The import order checking prevents subtle bugs where optimizations would be missed. This module establishes the foundation for Unsloth's 2x faster training by setting up optimizations at the earliest possible point.
