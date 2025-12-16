# File: `unsloth/kernels/moe/grouped_gemm/interface.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 968 |
| Classes | `GroupedGemm` |
| Functions | `supports_tma`, `get_per_device_per_stream_alloc_fn`, `log_kernel_info`, `grouped_gemm_forward`, `grouped_gemm_dX`, `grouped_gemm_dW`, `check_valid_config_fwd`, `check_valid_config_bwd_dW`, `check_valid_config_bwd_dX`, `grouped_gemm` |
| Imports | dataclasses, grouped_gemm, logging, torch, triton, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides the main user-facing interface for fused grouped GEMM operations in MoE layers, including forward pass and backward gradients (dX and dW).

**Mechanism:**
- Implements `grouped_gemm_forward`, `grouped_gemm_dX`, `grouped_gemm_dW` functions that wrap Triton kernels
- Supports MoE-specific optimizations: permute_x (gather), permute_y (scatter), fuse_mul_post (topk weights)
- Provides `GroupedGemm` autograd.Function for automatic differentiation
- Handles TMA (Tensor Memory Accelerator) support detection and configuration
- Validates kernel configurations to prevent invalid parameter combinations
- Manages memory allocation for Triton kernels with per-device allocators
- Logs kernel compilation info (registers, spills, best configs)

**Significance:** Core interface layer that bridges PyTorch autograd with optimized Triton kernels. Handles the complexity of MoE-specific fusions (permutations, weight merging) while maintaining compatibility with PyTorch's gradient computation. Essential for achieving high performance in MoE training and inference.