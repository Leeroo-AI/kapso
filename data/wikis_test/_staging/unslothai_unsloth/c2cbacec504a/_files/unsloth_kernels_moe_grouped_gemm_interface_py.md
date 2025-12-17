# File: `unsloth/kernels/moe/grouped_gemm/interface.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 968 |
| Classes | `GroupedGemm` |
| Functions | `supports_tma`, `get_per_device_per_stream_alloc_fn`, `log_kernel_info`, `grouped_gemm_forward`, `grouped_gemm_dX`, `grouped_gemm_dW`, `check_valid_config_fwd`, `check_valid_config_bwd_dW`, `... +2 more` |
| Imports | dataclasses, grouped_gemm, logging, torch, triton, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** High-level public interface for grouped GEMM operations with MoE fusions and autotuning support.

**Mechanism:** Wraps Triton kernels with configuration validation, TMA support detection, memory allocator setup, kernel compilation, and logging; defines grouped_gemm_forward, grouped_gemm_dX, grouped_gemm_dW functions for forward and backward passes.

**Significance:** User-facing API for grouped GEMM with full feature support and error handling - all MoE operations route through this interface.
