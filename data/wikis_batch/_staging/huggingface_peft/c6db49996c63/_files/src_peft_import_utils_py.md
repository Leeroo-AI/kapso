# File: `src/peft/import_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 172 |
| Functions | `is_bnb_available`, `is_bnb_4bit_available`, `is_auto_gptq_available`, `is_gptqmodel_available`, `is_optimum_available`, `is_torch_tpu_available`, `is_aqlm_available`, `is_auto_awq_available`, `... +6 more` |
| Imports | functools, importlib, packaging, platform, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides runtime detection of optional dependencies for quantization and hardware backends.

**Mechanism:** Each function checks if a specific library is available and meets minimum version requirements using importlib.util.find_spec and packaging.version. Functions are cached with @lru_cache to avoid repeated checks. Covers quantization libraries (bitsandbytes, auto-gptq, gptqmodel, aqlm, awq, eetq, hqq, torchao, inc) and hardware backends (TPU, XPU).

**Significance:** Essential for graceful degradation when optional dependencies are missing. Allows PEFT to support many quantization methods and hardware platforms without requiring all dependencies to be installed. These checks enable feature detection and appropriate fallbacks throughout the codebase.
