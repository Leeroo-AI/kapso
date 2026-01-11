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

**Purpose:** Main interface for grouped GEMM operations in MoE layers, providing forward and backward pass implementations with autograd integration.

**Mechanism:** Implements GroupedGemm autograd function with three kernel dispatch functions: grouped_gemm_forward (Y = X @ W), grouped_gemm_dX (gradient wrt inputs), and grouped_gemm_dW (gradient wrt weights). Supports permutation fusion (permute_x/permute_y), topk weight multiplication, TMA (Tensor Memory Accelerator) loads/stores for Hopper+ GPUs, and both manual tuning and autotuning modes. Handles routing indices for expert assignment and validates configurations for different GEMM positions in MoE blocks.

**Significance:** Core API for MoE acceleration. This is the primary entry point that transforms standard MoE computations into optimized grouped GEMM operations. The fusions (permutation, weight multiplication) eliminate memory-bound operations, while TMA support enables maximum bandwidth on modern GPUs. Critical for achieving competitive performance with frameworks like vLLM.
