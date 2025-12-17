# File: `unsloth/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 286 |
| Imports | chat_templates, functools, import_fixes, importlib, inspect, models, numpy, os, packaging, re, ... +8 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main package initialization that applies critical compatibility patches and imports all public APIs for the Unsloth optimization framework.

**Mechanism:** Imports are strategically ordered before loading trl, transformers, and peft libraries. It applies import-time patches via import_fixes module to address version compatibility issues, then conditionally loads bitsandbytes and triton for CUDA support. Sets environment variables and detects hardware capabilities (CUDA/HIP/XPU) to determine supported features.

**Significance:** Acts as the entry point for Unsloth users; import order is critical because this module patches libraries at import time before they can be used. Early import of unsloth vs trl/transformers determines whether optimizations are applied. Manages device capability detection and BFLOAT16 support checks.
