# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Functions | `_grouped_gemm_forward_kernel` (Triton JIT kernel) |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements the Triton kernel for the forward pass of grouped GEMM operations in MoE layers.

**Mechanism:**
- Computes Y = X @ W^T for multiple expert groups in a single kernel launch
- Processes tiles in parallel across streaming multiprocessors (SMs)
- Iterates over experts, with each expert having variable token counts (m_sizes)
- Supports MoE-specific fusions:
  - `permute_x`: Fused gather (loads X in permuted order from token→expert)
  - `permute_y`: Fused scatter (stores Y in permuted order from expert→token)
  - `fuse_mul_post`: Multiplies output by topk_weights in epilogue (inference only)
- Uses TMA (Tensor Memory Accelerator) for efficient loads/stores on Hopper+ GPUs
- Accumulates in configurable dtype (fp32 or TF32 for better performance)
- Includes autotuning via @triton.autotune with configs from autotuning.py

**Significance:** Core forward kernel that enables high-performance MoE inference and training. The fused permutations eliminate separate gather/scatter operations, saving memory bandwidth. The grouped GEMM approach processes all experts in a single kernel launch, reducing launch overhead compared to per-expert GEMMs.