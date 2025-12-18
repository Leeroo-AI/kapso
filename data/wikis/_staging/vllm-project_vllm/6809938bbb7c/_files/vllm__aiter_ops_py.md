# File: `vllm/_aiter_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1339 |
| Classes | `rocm_aiter_ops` |
| Functions | `is_aiter_found`, `is_aiter_found_and_supported`, `if_aiter_supported` |
| Imports | collections, functools, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** ROCm/AMD GPU optimized operations via AITER library integration.

**Mechanism:** Provides a comprehensive set of GPU operations optimized for AMD ROCm platform through the AITER library. The `rocm_aiter_ops` class contains over 100 methods for operations like attention, matrix multiplication, quantization, MoE layers, and normalization. It checks for AITER availability at runtime and provides fallback behavior. Functions like `is_aiter_found()` detect if the library is available, while `if_aiter_supported()` is a decorator that conditionally executes code based on AITER support. The file includes operations for various data types (FP8, FP16, INT4, INT8) and attention mechanisms (paged attention, flash attention, MLA).

**Significance:** Essential for AMD GPU support in vLLM. Enables high-performance inference on ROCm platforms by providing hardware-specific optimizations. The modular design allows vLLM to leverage AMD-specific optimizations while maintaining compatibility with other platforms. This is controlled by various `VLLM_ROCM_USE_AITER_*` environment variables.
