# File: `vllm/scalar_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 355 |
| Classes | `NanRepr`, `ScalarType`, `scalar_types` |
| Imports | dataclasses, enum, functools, struct |

## Understanding

**Status:** ✅ Explored

**Purpose:** Representation system for sub-byte and custom numeric types.

**Mechanism:** `ScalarType` is a frozen dataclass representing arbitrary numeric types including sub-byte formats. Key fields: `exponent` and `mantissa` bits for floats, `signed` flag, `bias` for offset encoding, `_finite_values_only`, and `nan_repr` enum (IEEE_754, EXTD_RANGE_MAX_MIN, NONE). Provides methods to compute min/max values, check properties (is_floating_point, is_signed), and convert to/from integer IDs for PyTorch custom ops. The `scalar_types` class defines common types: int4/8, uint4/8, float8_e4m3fn, float8_e5m2, float6/4 variants, and GPTQ biased types (uint2b2, uint4b8, etc.). Naming follows ml_dtypes conventions.

**Significance:** Essential for quantization support. Enables vLLM to work with FP8, FP6, FP4, INT4, and other low-precision formats that PyTorch doesn't natively support. The bias field is crucial for GPTQ-style quantization. This type system bridges Python-level configuration with C++/CUDA kernel implementations. Proper handling of exotic types is critical for supporting modern quantization methods that achieve high compression with minimal quality loss.
