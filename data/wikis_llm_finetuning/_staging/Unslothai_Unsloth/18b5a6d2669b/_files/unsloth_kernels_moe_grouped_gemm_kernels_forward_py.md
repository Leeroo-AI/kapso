# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel implementation for grouped GEMM forward pass (Y = X @ W.T) with MoE-specific fusions.

**Mechanism:** Implements _grouped_gemm_forward_kernel that iterates over experts, processing tokens in tiles. For each expert, loads input tiles (with optional permutation via gather_indices), weight tiles, performs matrix multiplication with tensor cores, and stores results (with optional scatter back to token order). Supports TMA loads/stores on Hopper+ GPUs, fused topk weight multiplication in epilogue, and @triton.autotune for configuration selection. Uses persistent work distribution across streaming multiprocessors.

**Significance:** The core computational kernel for MoE inference and training. Performance here directly impacts end-to-end model throughput. The permutation and weight multiplication fusions eliminate separate kernel launches, reducing memory traffic. TMA support achieves near-peak memory bandwidth on modern GPUs. This kernel is what enables Unsloth to match or exceed specialized MoE frameworks.
