# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel implementation for the forward pass of grouped GEMM in MoE layers, computing X @ W^T for multiple expert groups efficiently.

**Mechanism:** The `_grouped_gemm_forward_kernel` iterates over experts, computing tiles of the output matrix Y = X @ W^T. Key features include: PERMUTE_X fuses token permutation from token order to expert order during load, PERMUTE_Y fuses output permutation back to token order during store, FUSE_MUL_PRE/POST multiplies by topk weights within the kernel. Uses TMA descriptors (`tl._experimental_make_tensor_descriptor`) for efficient weight and activation loading on SM90+ GPUs. Employs persistent kernel pattern where thread blocks loop through tiles across all experts. Block-based tiling with configurable M/N/K block sizes enables memory hierarchy optimization.

**Significance:** Core forward computation kernel for MoE models - enables efficient batched matrix multiplication across multiple experts with fused permutation and weight multiplication operations.
