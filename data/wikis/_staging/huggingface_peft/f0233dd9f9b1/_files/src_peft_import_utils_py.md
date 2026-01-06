# File: `src/peft/import_utils.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 172 |
| Functions | `is_bnb_available`, `is_bnb_4bit_available`, `is_auto_gptq_available`, `is_gptqmodel_available`, `is_optimum_available`, `is_torch_tpu_available`, `is_aqlm_available`, `is_auto_awq_available`, `is_eetq_available`, `is_hqq_available`, `is_inc_available`, `is_torchao_available`, `is_xpu_available`, `is_diffusers_available` |
| Imports | functools, importlib, packaging, platform, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Lazy availability checks for optional dependencies.

**Mechanism:** Provides `@lru_cache` decorated functions checking if optional packages are installed: bitsandbytes (8/4-bit quantization), auto_gptq/gptqmodel (GPTQ quantization), awq (AWQ quantization), aqlm, eetq, hqq (Half-Quadratic Quantization), neural_compressor (Intel), torchao, diffusers, torch_xla (TPU). Includes version checks with minimum version requirements (e.g., auto_gptq>=0.5.0, torchao>=0.4.0). XPU check handles Intel GPU availability.

**Significance:** Enables graceful degradation when optional backends unavailable. Quantization methods (bnb, gptq, awq) are conditionally loaded based on these checks.
