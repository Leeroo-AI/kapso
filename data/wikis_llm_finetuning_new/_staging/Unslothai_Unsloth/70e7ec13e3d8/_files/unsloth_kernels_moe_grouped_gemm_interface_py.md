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

**Purpose:** Main interface module providing the `grouped_gemm` function and `GroupedGemm` autograd class for MoE grouped matrix multiplications with forward and backward pass support.

**Mechanism:** Implements `grouped_gemm_forward`, `grouped_gemm_dX`, and `grouped_gemm_dW` functions that dispatch to either autotuned or manually-configured Triton kernels. Supports TMA (Tensor Memory Accelerator) loading on SM90+ GPUs via `supports_tma()` check. The `GroupedGemm` class extends `torch.autograd.Function` for gradient computation. Key features include fused permutation (permute_x/permute_y) to reorder tokens between token and expert order, fused topk weight multiplication, and configurable block sizes. Includes validation functions `check_valid_config_*` to ensure TMA and permutation options are compatible.

**Significance:** Core API module that bridges high-level MoE layer calls to optimized Triton kernels, forming the primary entry point for using grouped GEMM in training and inference.
