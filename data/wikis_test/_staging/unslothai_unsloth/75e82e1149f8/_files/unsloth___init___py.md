# File: `unsloth/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 280 |
| Imports | chat_templates, functools, import_fixes, importlib, inspect, models, numpy, os, packaging, re, ... +8 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main package entry point that initializes Unsloth and applies all optimizations before any other libraries load.

**Mechanism:**
- Sets `UNSLOTH_IS_PRESENT` environment variable
- Checks if critical modules (trl, transformers, peft) are already imported and warns if so (optimizations may not apply)
- Applies compatibility fixes via `import_fixes` module (protobuf, xformers, vLLM, datasets)
- Detects device type (CUDA/HIP/XPU) and configures bfloat16 support
- Links CUDA libraries via ldconfig if bitsandbytes fails to load
- Imports and re-exports all public APIs from submodules (models, save, chat_templates, tokenizer_utils, trainer)
- Patches TRL trainers for backwards compatibility

**Significance:** This is the critical entry point - Unsloth MUST be imported before other ML libraries to apply its monkey-patches and optimizations. The import order check prevents silent performance regressions.
